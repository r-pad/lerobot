"""ScriptRobot: direct scripted Franka motion without a GELLO leader."""

import json
import os
import time

import numpy as np
import torch
import pytorch3d.transforms as transforms
from scipy.spatial.transform import Rotation as R

from lerobot.common.robot_devices.control_utils import (
    compute_foundation_stereo_depth,
    droid_ik_model_eef_pose,
    get_zed_intrinsics_and_baseline,
    print_droid_ik_diagnostics,
    read_zed_stereo_rgb,
)
from lerobot.common.robot_devices.robots.configs import ScriptRobotConfig
from lerobot.common.robot_devices.robots.droid import DroidRobot
from lerobot.common.robot_devices.robots.robot_ik_controller import RobotIKController
from lerobot.common.utils.pointcloud_rgbd import render_top_down_custom
from lerobot.common.policy.force_diffusion_policy_image_condition import Diffusion_Policy


WRIST_CAM_TO_GRIPPER = np.array(
    [
        [-0.00768086, -0.94557934, -0.32530096, 0.07294499],
        [0.99995759, -0.00891583, 0.00230583, -0.03177615],
        [-0.00508067, -0.32526946, 0.94560772, -0.08727812],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
import h5py
def downsample_depth_with_top_padding(depth_img, target_h=360, target_w=640):
    """Downsample, pad top/width to target size, normalize, rotate 180 degrees, and negate."""
    # Downsample by nearest-neighbor index selection.
    row_idx = np.linspace(0, depth_img.shape[0] - 1, target_h).astype(np.int64)
    col_idx = np.linspace(0, depth_img.shape[1] - 1, target_w).astype(np.int64)
    resized = depth_img[row_idx][:, col_idx]
    # If a later change makes height smaller than target, pad above with the top row.
    if resized.shape[0] < 480:
        pad_top = 480 - resized.shape[0]
        top_row = resized[0:1, :]
        top_pad = np.repeat(top_row, pad_top, axis=0)
        resized = np.concatenate([top_pad, resized], axis=0)

    if resized.shape[1] < target_w:
        pad_w = target_w - resized.shape[1]
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        left_col = resized[:, 0:1]
        right_col = resized[:, -1:]
        left_pad = np.repeat(left_col, pad_left, axis=1)
        right_pad = np.repeat(right_col, pad_right, axis=1)
        resized = np.concatenate([left_pad, resized, right_pad], axis=1)
    resized = - resized
    resized = resized.astype(np.float32, copy=False)
    resized = np.clip(resized, a_min=-0.5, a_max=0.0)
    resized = (resized - (-0.1890)) / 0.0795
    return np.rot90(resized, 2)
def unnormalize_actions(actions, pos_min, pos_max, rot_min,rot_max, scale_to_unit=True):
    """
    Undo the normalization from DepthActionDataset.
    
    Args:
        actions: (B, K, 9) torch.Tensor, with normalized delta_pos and raw rot6d
        pos_min: np.ndarray or torch.Tensor, shape (3,)
        pos_max: np.ndarray or torch.Tensor, shape (3,)
        scale_to_unit: bool, whether dataset was scaled to [-1, 1] or just [0, 1]
    Returns:
        unnorm_actions: (B, K, 9) torch.Tensor, with real delta_pos + rot6d
    """
    # device = actions.device
    # pos_min = torch.tensor(pos_min, dtype=torch.float32, device=device)
    # pos_max = torch.tensor(pos_max, dtype=torch.float32, device=device)

    delta_norm = actions[..., :3]
    delta_norm_rot=actions[...,3:]
    if scale_to_unit:
        # [-1,1] -> [0,1]
        delta_norm = (delta_norm + 1.0) / 2.0
        delta_norm_rot=(delta_norm_rot + 1.0) / 2.0
    # [0,1] -> original range
    delta_pos = delta_norm * (pos_max - pos_min) + pos_min
    delta_rot = delta_norm_rot * (rot_max-rot_min) + rot_min
    return torch.cat([delta_pos, delta_rot], dim=-1)

def load_processed_dataset(filename):
    import h5py
    data = {}
    with h5py.File(filename, "r") as f:
        # data["depth"] = f["depth"][()]               # (N, H, W)
        data["proprioception"] = f["proprioception"][()]  # (N, 9)
        data["actions"] = f["actions"][()]           # (N, K, 9)
        data["force"] = f["force"][()]             # (N, K, 3)
    return data

class _FrankaInterfaceControlAdapter:
    """Delegate FrankaInterface calls while tolerating mentor controller kwargs."""

    def __init__(self, robot_interface):
        self._robot_interface = robot_interface

    def __getattr__(self, name):
        return getattr(self._robot_interface, name)

    def control(self, *args, **kwargs):
        kwargs.pop("binary_grasping", None)
        result = self._robot_interface.control(*args, **kwargs)
        time.sleep(0.01)
        return result


def _visualize_depth_point_cloud_once(depth: np.ndarray, rgb: np.ndarray, k: np.ndarray, max_depth_m: float = 0.5):
    import open3d as o3d

    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape[:2]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth
    x = (xx.astype(np.float32) - float(k[0, 2])) * z / float(k[0, 0])
    y = (yy.astype(np.float32) - float(k[1, 2])) * z / float(k[1, 1])
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3).astype(np.float64)
    if colors.max() > 1:
        colors /= 255.0

    keep = np.isfinite(points).all(axis=1) & (points[:, 2] > 0) & (points[:, 2] <= max_depth_m)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[keep].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors[keep])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="First wrist FoundationStereo point cloud")
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    vis.run()
    vis.destroy_window()


class ScriptRobot(DroidRobot):
    """Droid-like Franka robot that executes a built-in scripted motion.

    Each control step lifts the current end-effector target along world Z by
    ``config.z_step`` metres, converts that EEF target to joint space, and sends
    it through the same deoxys joint controller used by DroidRobot.
    """

    robot_type = "script"

    def __init__(self, config: ScriptRobotConfig | None = None, **kwargs):
        super().__init__(config if config is not None else ScriptRobotConfig(**kwargs))
        self._script_step_count = 0
        self._script_joint_target = None
        self._home_move_done = False
        self._pose_move_done = False
        self._pose_target_pos = None
        self._pose_target_rot = None
        self._pose_target_origin_pos = None
        self._last_script_delta_pos = np.zeros(3, dtype=np.float64)
        self._recorded_insertion_pose = None
        self._robot_ik_controller = None
        self._last_joint_target = None
        self._last_joint_target_reached = True
        self._teleop_hold_z = None
        self.force_buffer = None
        self._last_gripper_action = self.config.gripper_close_action
        data=load_processed_dataset("/home/yinongh/automate/lerobot/ckpt/processed_dataset_forward_sim2real_0523_automate.h5")
        all_actions = data['actions'][()]  # (N, K, 9)
        delta_pos = all_actions[..., 0:3]  # (N, K, 3)
        delta_rot = all_actions[...,3:]
        self.pos_min = torch.from_numpy(delta_pos.min(axis=(0, 1))).cuda()
        self.pos_max = torch.from_numpy(delta_pos.max(axis=(0, 1))).cuda()
        self.rot_min=torch.from_numpy(delta_rot.min(axis=(0,1))).cuda()
        self.rot_max=torch.from_numpy(delta_rot.max(axis=(0,1))).cuda()
        self.rot_min=torch.from_numpy(delta_rot.min(axis=(0,1))).cuda()
        self.rot_max=torch.from_numpy(delta_rot.max(axis=(0,1))).cuda()
        self.force_min = np.min(data['force'][()], axis=0)
        self.force_max = np.max(data['force'][()], axis=0)
        self.policy = Diffusion_Policy(
            action_dim=9,
            obs_feature_dim= 512,
            hidden_dim=512,
            num_action = 10,
        ).to("cuda")
        state_dict = torch.load("/home/yinongh/automate/lerobot/ckpt/real_world_2/automate_policy_epoch_200.ckpt", map_location="cuda")
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    @property
    def camera_features(self) -> dict:
        camera_features = {}
        if "cam_wrist" in self.cameras:
            cam_cfg = self.cameras["cam_wrist"].config
            camera_features["observation.images.cam_wrist"] = {
                "shape": (cam_cfg.height, cam_cfg.width, cam_cfg.channels),
                "names": ["height", "width", "channels"],
                "info": f"{cam_cfg.color_mode.upper()} color image",
            }
        for cam_key, cam in self.cameras.items():
            if cam_key.startswith("cam_azure"):
                camera_features.update(cam.config.get_feature_specs(cam_key))
        return camera_features

    @property
    def motor_features(self) -> dict:
        motor_features = {
            "action": {
                "dtype": "float32",
                "shape": (9,),
                "names": [
                    "delta_x",
                    "delta_y",
                    "delta_z",
                    "delta_rot6d_0",
                    "delta_rot6d_1",
                    "delta_rot6d_2",
                    "delta_rot6d_3",
                    "delta_rot6d_4",
                    "delta_rot6d_5",
                ],
            },
            "observation.eef_internal_forces": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["isaacgym_cam_wrist_fx", "isaacgym_cam_wrist_fy", "isaacgym_cam_wrist_fz"],
                "info": "Linear EEF internal force mapped into the wrist camera frame with IsaacGym x/z axis convention",
            },
            "observation.eef_pose": {
                "dtype": "float32",
                "shape": (4, 4),
                "names": ["row", "col"],
            },
            "observation.aligned_socket_depth_img": {
                "dtype": "float32",
                "shape": (480, 640),
                "names": ["height", "width"],
                "info": "Socket depth image aligned by rotating opposite current EEF yaw change",
            },
            "observation.init_plug_depth_img": {
                "dtype": "float32",
                "shape": (480, 640),
                "names": ["height", "width"],
                "info": "Initial plug depth image rendered from the auxiliary point cloud",
            },
            "observation.points.initial_wrist_points_world": {
                "dtype": "pcd",
                "shape": (-1, 3),
                "names": ["points", "xyz"],
            },
        }
        if self.config.debug:
            # teleop_step only produces these depth images when running policy inference.
            motor_features.pop("observation.aligned_socket_depth_img")
            motor_features.pop("observation.init_plug_depth_img")
        if "cam_wrist" in self.cameras:
            motor_features["observation.cam_wrist.extrinsics"] = {
                "dtype": "float32",
                "shape": (4, 4),
                "names": ["rows", "cols"],
                "info": "Wrist camera extrinsic matrix (T_world_cam)",
            }
            cam_cfg = self.cameras["cam_wrist"].config
            motor_features["observation.images.cam_wrist.depth"] = {
                "dtype": "float32",
                "shape": (cam_cfg.height, cam_cfg.width),
                "names": ["height", "width"],
                "info": "FoundationStereo depth from wrist ZED Mini in meters",
            }
        return motor_features

    def _wrist_camera_extrinsics(self, eef_pose: np.ndarray) -> np.ndarray:
        world_from_gripper = np.asarray(eef_pose, dtype=np.float32).reshape(4, 4)
        return (world_from_gripper @ WRIST_CAM_TO_GRIPPER).astype(np.float32)

    def _world_vector_to_wrist_camera(
        self,
        vector_world: np.ndarray,
        wrist_extrinsics: np.ndarray,
    ) -> np.ndarray:
        world_from_cam = np.asarray(wrist_extrinsics, dtype=np.float32).reshape(4, 4)
        vector_cam = world_from_cam[:3, :3].T @ np.asarray(vector_world, dtype=np.float32).reshape(3)
        return vector_cam.astype(np.float32)

    def _wrist_camera_vector_to_world(
        self,
        vector_cam_wrist: np.ndarray,
        wrist_extrinsics: np.ndarray,
    ) -> np.ndarray:
        world_from_cam = np.asarray(wrist_extrinsics, dtype=np.float32).reshape(4, 4)
        vector_world = world_from_cam[:3, :3] @ np.asarray(vector_cam_wrist, dtype=np.float32).reshape(3)
        return vector_world.astype(np.float32)

    def _wrist_camera_vector_to_isaacgym(self, vector_cam_wrist: np.ndarray) -> np.ndarray:
        vector_isaacgym = np.asarray(vector_cam_wrist, dtype=np.float32).reshape(3).copy()
        vector_isaacgym[[0, 2]] *= -1.0
        return vector_isaacgym

    def _world_vector_to_isaacgym_wrist_camera(
        self,
        vector_world: np.ndarray,
        wrist_extrinsics: np.ndarray,
    ) -> np.ndarray:
        vector_cam_wrist = self._world_vector_to_wrist_camera(vector_world, wrist_extrinsics)
        return self._wrist_camera_vector_to_isaacgym(vector_cam_wrist)

    def _isaacgym_wrist_camera_vector_to_world(
        self,
        vector_isaacgym_wrist: np.ndarray,
        wrist_extrinsics: np.ndarray,
    ) -> np.ndarray:
        vector_cam_wrist = self._wrist_camera_vector_to_isaacgym(vector_isaacgym_wrist)
        return self._wrist_camera_vector_to_world(vector_cam_wrist, wrist_extrinsics)


    def _relative_world_yaw(self, init_eef_pose: np.ndarray, current_eef_pose: np.ndarray) -> float:
        init_rot = np.asarray(init_eef_pose, dtype=np.float32).reshape(4, 4)[:3, :3]
        current_rot = np.asarray(current_eef_pose, dtype=np.float32).reshape(4, 4)[:3, :3]
        relative_rot = current_rot @ init_rot.T
        return float(np.arctan2(relative_rot[1, 0], relative_rot[0, 0]))

    def _control_refined(self, target_pos, target_rot, max_iters: int | None = None, verbose: bool = True) -> bool:
        """Move to the full 6-DoF target pose with closed-loop refinement (see RobotIKController.control_refined)."""
        cfg = self.config
        return self._robot_ik_controller.control_refined(
            target_pos=target_pos,
            target_rot=target_rot,
            grasping_action=getattr(self, "_last_gripper_action", cfg.gripper_open_action),
            max_iters=cfg.pose_refine_max_iters if max_iters is None else max_iters,
            pos_tol=cfg.pose_refine_pos_tol,
            rot_tol_deg=cfg.pose_refine_rot_tol_deg,
            gain=cfg.pose_refine_gain,
            settle_steps=cfg.pose_refine_settle_steps,
            max_pos_correction=cfg.pose_refine_max_pos_correction,
            max_rot_correction_deg=cfg.pose_refine_max_rot_correction_deg,
            wait_times=50,
            joint_threshold=float(cfg.script_joint_solution_threshold),
            verbose=verbose,
        )

    def _rotate_image(self, image: np.ndarray | torch.Tensor, angle_deg: float, output_shape=(480, 640)) -> np.ndarray:
        import cv2

        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]

        h, w = image.shape[:2]
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
        rotated = cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        if output_shape is not None and rotated.shape[:2] != tuple(output_shape):
            rotated = cv2.resize(rotated, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_LINEAR)
        return rotated.astype(np.float32)

    def _rotate_socket_depth_opposite_eef_yaw(
        self,
        init_socket_depth_img: np.ndarray | torch.Tensor,
        init_eef_pose: np.ndarray,
        current_eef_pose: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        delta_yaw = self._relative_world_yaw(init_eef_pose, current_eef_pose)
        delta_yaw_deg = float(np.rad2deg(delta_yaw))
        rotated_depth = self._rotate_image(init_socket_depth_img, -delta_yaw_deg, output_shape=(480, 640))
        print(
            f"[script] Rotating init socket depth by {-delta_yaw_deg:.2f} deg "
            f"to compensate EEF yaw change of {delta_yaw_deg:.2f} deg"
        )
        return rotated_depth, delta_yaw_deg

    def _find_robotiq_port_without_gello(self) -> str:
        import minimalmodbus as mm
        import serial
        import serial.tools.list_ports

        for port_info in serial.tools.list_ports.comports():
            try:
                ser = serial.Serial(port_info.device, 115200, 8, "N", 1, 0.2)
                device = mm.Instrument(ser, 9, mm.MODE_RTU, close_port_after_each_call=False, debug=False)
                device.write_registers(1000, [0, 100, 0])
                registers = device.read_registers(2000, 3, 4)
                echo = registers[1] & 0xFF
                del device
                ser.close()
                if echo == 100:
                    print(f"Robotiq gripper found on {port_info.device}")
                    return port_info.device
            except Exception:
                continue

        raise RuntimeError(
            "No Robotiq gripper found. Please specify robotiq_port in the config, "
            "or check that the gripper is connected."
        )
    def _init_robotiq_gripper_without_gello(self):
        from pyrobotiqgripper import RobotiqGripper

        port = self.config.robotiq_port
        if port is None:
            port = self._find_robotiq_port_without_gello()

        print(f"Connecting to Robotiq gripper on {port}")
        self.robotiq_gripper = RobotiqGripper(portname=port)
        print("Activating Robotiq gripper (will fully open/close during activation)...")
        self.robotiq_gripper.activate()
        print("Robotiq gripper activated.")
    def _get_eef_internal_forces(
        self,
        eef_pose: np.ndarray,
        wrist_extrinsics: np.ndarray,
    ) -> torch.Tensor:
        if self.robot_interface.state_buffer_size == 0:
            return torch.zeros(3, dtype=torch.float32)

        state = self.robot_interface._state_buffer[-1]
        if hasattr(state, "O_F_ext_hat_K"):
            wrench = np.asarray(state.O_F_ext_hat_K, dtype=np.float32).reshape(-1)
            if wrench.size >= 3:
                force_world = wrench[:3]
                force_isaacgym_cam_wrist = self._world_vector_to_isaacgym_wrist_camera(
                    force_world,
                    wrist_extrinsics,
                )
                return torch.from_numpy(force_isaacgym_cam_wrist)

        if hasattr(state, "K_F_ext_hat_K"):
            wrench = np.asarray(state.K_F_ext_hat_K, dtype=np.float32).reshape(-1)
            if wrench.size >= 3:
                world_from_k = np.asarray(eef_pose, dtype=np.float32).reshape(4, 4)[:3, :3]
                force_world = world_from_k @ wrench[:3]
                force_isaacgym_cam_wrist = self._world_vector_to_isaacgym_wrist_camera(
                    force_world,
                    wrist_extrinsics,
                )
                return torch.from_numpy(force_isaacgym_cam_wrist)

        return torch.zeros(3, dtype=torch.float32)

    def connect(self):
        if self.is_connected:
            raise RuntimeError("ScriptRobot is already connected. Do not run `robot.connect()` twice.")

        from deoxys.franka_interface import FrankaInterface
        from deoxys.utils import YamlConfig

        self.robot_interface = FrankaInterface(
            self.config.deoxys_general_cfg_file,
            use_visualizer=False,
            has_gripper=False,
            automatic_gripper_reset=True,
        )
        self.controller_cfg = YamlConfig(
            self.config.deoxys_controller_cfg_file
        ).as_easydict()

        print("Waiting for Franka state buffer...")
        timeout = 30.0
        start_t = time.time()
        while len(self.robot_interface._state_buffer) == 0:
            time.sleep(0.1)
            if time.time() - start_t > timeout:
                raise TimeoutError(
                    "Timed out waiting for Franka state buffer. "
                    "Check that the deoxys controller is running."
                )
        print("Franka state buffer ready.")

        # self._init_robotiq_gripper_without_gello()
        self._connect_cameras()
        self._load_recorded_insertion_pose()
        self._init_robot_ik_controller()

        self.is_connected = True
        print(
            "[ScriptRobot] Connected. "
            f"script_mode={self.config.script_mode} controller={self.config.deoxys_controller_type}"
        )

    def _init_robot_ik_controller(self):
        try:
            self._robot_ik_controller = RobotIKController(
                # impedance_control=self.config.deoxys_controller_type == "JOINT_IMPEDANCE",
                impedance_control=False,
                use_bullet=True,
                binary_grasping=False,
                robot_interface=_FrankaInterfaceControlAdapter(self.robot_interface),
                controller_cfg=self.controller_cfg,
                controller_type=self.config.deoxys_controller_type,
            )
            print(
                "[ScriptRobot] RobotIKController ready. "
                f"urdf={self._robot_ik_controller.bullet_ik_wrapper.urdf_path} "
                f"eef_idx={self._robot_ik_controller.bullet_ik_wrapper.right_eef_idx}"
            )
        except Exception as exc:
            self._robot_ik_controller = None
            raise RuntimeError(
                "RobotIKController is required for ScriptRobot PyBullet IK, but it failed to initialize. "
                "Fix this first; ScriptRobot will not fall back to MuJoCo IK."
            ) from exc

    def _pybullet_current_eef_pose(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._robot_ik_controller is None:
            raise RuntimeError("RobotIKController is not initialized; cannot get PyBullet EEF pose.")

        pos, quat_xyzw = self._robot_ik_controller.bullet_ik_wrapper.forward_kinematics(joints)
        rot = R.from_quat(quat_xyzw).as_matrix()
        return rot, np.asarray(pos, dtype=np.float64)


    def _connect_cameras(self):
        from threading import Thread

        azure_kinect_cameras = []
        for name, camera in self.cameras.items():
            if camera.__class__.__name__ == "AzureKinectCamera":
                camera.connect(start_cameras=False)
                azure_kinect_cameras.append(camera)
            else:
                camera.connect()

        if len(azure_kinect_cameras) > 0:
            def start_camera(cam):
                cam.start()

            threads = [Thread(target=start_camera, args=(cam,)) for cam in azure_kinect_cameras]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def run_calibration(self):
        print("[ScriptRobot] Skipping GELLO calibration; scripted control starts from the current Franka pose.")

    def open_gripper(self):
        print("[ScriptRobot] Ignoring recorder open_gripper(); gripper is controlled by the script.")

    def _camera_output_to_tensors(self, camera_output):
        if isinstance(camera_output, dict):
            return {
                img_name: torch.from_numpy(image)
                for img_name, image in camera_output.items()
            }

        if isinstance(camera_output, tuple):
            if len(camera_output) != 2:
                raise ValueError(f"Expected camera tuple to contain (color, depth), got {len(camera_output)} items.")

            color_image, depth_map = camera_output
            if depth_map.ndim == 2:
                depth_map = depth_map[..., None]
            if np.issubdtype(depth_map.dtype, np.floating):
                depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=65535.0, neginf=0.0)
                depth_map = np.clip(depth_map, 0, 65535).astype(np.uint16)

            return {
                "color": torch.from_numpy(color_image),
                "depth": torch.from_numpy(depth_map),
            }

        return torch.from_numpy(camera_output)

    def _current_eef_pose(self, state: torch.Tensor) -> torch.Tensor:
        eef_rot, eef_pos = self.robot_interface.last_eef_rot_and_pos
        rot_6d = transforms.matrix_to_rotation_6d(torch.from_numpy(eef_rot[None])).squeeze()
        trans = torch.from_numpy(eef_pos.squeeze())
        return torch.cat([rot_6d, trans, state[-1:]], dim=0).float()

    def _home_joint_action(self, state: torch.Tensor) -> torch.Tensor:
        home = torch.tensor(self.config.home_joints, dtype=state.dtype, device=state.device)
        if home.numel() != 7:
            raise ValueError(f"Expected 7 home joints, got {home.numel()}: {self.config.home_joints}")

        action = state.clone()
        if self._script_joint_target is None:
            self._script_joint_target = state[:7].clone()

        delta = home - self._script_joint_target
        max_step = abs(float(self.config.max_joint_step))
        if max_step > 0:
            delta = delta.clamp(min=-max_step, max=max_step)
            self._script_joint_target = self._script_joint_target + delta
        else:
            self._script_joint_target = home.clone()

        action[:7] = self._script_joint_target
        action[7] = self.config.gripper_open_action
        return action

    def _smooth_move(self, target_joints: np.ndarray, gripper_action: float) -> tuple[float, int]:
        max_delta, num_steps = self._smooth_move_to(target_joints, gripper_action=gripper_action)
        if gripper_action == self.config.gripper_close_action:
            print("[script] Closing Robotiq gripper.")
            self.robotiq_gripper.close()
        else:
            print("[script] Opening Robotiq gripper.")
            self.robotiq_gripper.open()
        self._last_gripper_action = gripper_action

        return max_delta, num_steps

    def _load_recorded_insertion_pose(self):
        path = self.config.insertion_pose_path
        if not path or not os.path.exists(path):
            print(f"[ScriptRobot] No recorded insertion pose found at {path}")
            self._recorded_insertion_pose = None
            return

        with open(path) as f:
            self._recorded_insertion_pose = json.load(f)
        pose = self._recorded_insertion_pose["ik_grip_site"]
        print(
            "[ScriptRobot] Loaded insertion pose: "
            f"ik_pos={np.round(pose['position'], 6).tolist()}"
        )

    def _compute_insert_action(self, delta_pos: torch.Tensor) -> torch.Tensor:
        """Compute an insertion delta action from target-current position error."""
        action = torch.zeros_like(delta_pos)

        delta_xy = delta_pos[:, :2]
        delta_xy_norm = torch.norm(delta_xy, dim=1, keepdim=True)

        tmp = delta_pos.clone()
        tmp[:, 2] = 0.0
        delta_norm = torch.norm(tmp, dim=1, keepdim=True) + 1e-8

        mask1 = delta_xy_norm[:, 0] > 0.002
        mask2 = (delta_xy_norm[:, 0] <= 0.002) & (delta_xy_norm[:, 0] > 0.001)
        mask3 = (delta_xy_norm[:, 0] <= 0.001) & (delta_xy_norm[:, 0] > 0.0003)
        mask4 = (delta_xy_norm[:, 0] <= 0.0003) & (delta_xy_norm[:, 0] > 0.0001)
        mask5 = delta_xy_norm[:, 0] < 0.0001

        action[mask1] = tmp[mask1] / delta_norm[mask1] * 0.0003
        action[mask2] = tmp[mask2] / delta_norm[mask2] * 0.00015
        action[mask3] = tmp[mask3] / 5
        action[mask3, 2] = -0.0001
        action[mask4] = tmp[mask4]
        action[mask4, 2] = -0.0002
        action[mask5] = tmp[mask5]
        action[mask5, 2] = -0.0004

        # For now alignment is XY-only; keep height fixed.
        action[:, 2] = 0.0
        return action

    def _recorded_insertion_delta_pos(self, state: torch.Tensor) -> np.ndarray:
        if self._recorded_insertion_pose is None:
            self._load_recorded_insertion_pose()
        if self._recorded_insertion_pose is None:
            return np.zeros(3, dtype=np.float64)

        _, current_pos = droid_ik_model_eef_pose(state[:7].detach().cpu().numpy())
        target_pos = np.asarray(self._recorded_insertion_pose["ik_grip_site"]["position"], dtype=np.float64)
        delta_pos = torch.as_tensor((target_pos - current_pos)[None], dtype=torch.float32)
        action = self._compute_insert_action(delta_pos).squeeze(0).numpy().astype(np.float64)
        if self._script_step_count % 30 == 0:
            _, robot_pos = self.robot_interface.last_eef_rot_and_pos
            robot_pos = robot_pos.squeeze()
            live_frame_offset = current_pos - robot_pos
            recorded_robot_pos = np.asarray(
                self._recorded_insertion_pose.get("robot_eef", {}).get("position", target_pos),
                dtype=np.float64,
            )
            recorded_frame_offset = target_pos - recorded_robot_pos
            target_robot_pos = target_pos - live_frame_offset
            print(
                "[ScriptRobot] insert delta action\n"
                f"  ik_current={np.round(current_pos, 6).tolist()} "
                f"ik_target={np.round(target_pos, 6).tolist()} "
                f"ik_error={np.round(target_pos - current_pos, 6).tolist()} "
                f"ik_action={np.round(action, 6).tolist()}\n"
                f"  robot_current={np.round(robot_pos, 6).tolist()} "
                f"robot_target_from_ik={np.round(target_robot_pos, 6).tolist()} "
                f"robot_error={np.round(target_robot_pos - robot_pos, 6).tolist()}\n"
                f"  live_offset_ik_minus_robot={np.round(live_frame_offset, 6).tolist()} "
                f"recorded_offset_ik_minus_robot={np.round(recorded_frame_offset, 6).tolist()}"
            )
        return action

    def _script_delta_pos(self, state: torch.Tensor) -> np.ndarray:
        """Return the Cartesian delta command for this scripted step."""
        # if self.config.script_mode == "move_to_insertion":
        #     return self._recorded_insertion_delta_pos(state)

        # delta_pos = np.asarray(self.config.script_delta_pos, dtype=np.float64)
        # delta_pos[2] = 0.0
        delta_pos = np.asarray([0,0,0.001],dtype=np.float64)
        return delta_pos

    def compute_insert_action(self, delta_pos):
        """
        delta_pos: (3,) tensor
        return:
            action: (3,)
        """
        action = torch.zeros_like(delta_pos)

        delta_xy = delta_pos[:2]
        delta_xy_norm = torch.norm(delta_xy)

        tmp = delta_pos.clone()
        tmp[2] = 0

        delta_norm = torch.norm(tmp) + 1e-8

        if delta_xy_norm > 0.002:
            action = tmp / delta_norm * 0.0003

        elif delta_xy_norm > 0.001:
            action = tmp / delta_norm * 0.00015

        elif delta_xy_norm > 0.0003:
            action = tmp / 5
            action[2] = -0.0001

        elif delta_xy_norm > 0.0001:
            action = tmp
            action[2] = -0.0002

        else:
            action = tmp
            action[2] = -0.0004

        return action

    def _interpolate_rotation_matrix(
        self,
        current_rot: np.ndarray,
        aligned_rot: np.ndarray,
        max_angle_step_deg: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return one bounded quaternion step from current_rot toward aligned_rot.

        Quaternion format is xyzw throughout. scipy Rotation.as_quat() returns xyzw,
        and RobotIKController receives the returned rotation matrix.
        """
        if max_angle_step_deg is None:
            max_angle_step_deg = float(getattr(self.config, "script_rot_max_angle_step_deg", 0.3))

        current_q = torch.as_tensor(
            R.from_matrix(np.asarray(current_rot, dtype=np.float64)).as_quat(),
            dtype=torch.float64,
        )[None]
        target_q = torch.as_tensor(
            R.from_matrix(np.asarray(aligned_rot, dtype=np.float64)).as_quat(),
            dtype=torch.float64,
        )[None]
        if torch.sum(current_q * target_q, dim=-1).item() < 0.0:
            target_q = -target_q

        q_step, _ = self._quat_step_toward(
            current_q=current_q,
            target_q=target_q,
            max_angle_step_deg=max_angle_step_deg,
        )
        next_q = self._quat_mul(q_step, current_q)
        next_q = self._quat_normalize(next_q)
        target_rot = R.from_quat(next_q.squeeze(0).numpy()).as_matrix()
        return target_rot, current_q.squeeze(0).numpy(), target_q.squeeze(0).numpy(), next_q.squeeze(0).numpy()

    @staticmethod
    def _quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=eps)

    @staticmethod
    def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        qc = q.clone()
        qc[:, :3] = -qc[:, :3]
        return qc

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        # Quaternion format: xyzw.
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        return torch.stack(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dim=-1,
        )

    @classmethod
    def _quat_to_axis_angle(cls, q: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
        q = cls._quat_normalize(q)

        xyz = q[:, :3]
        w = torch.clamp(q[:, 3], -1.0, 1.0)

        sin_half = torch.norm(xyz, dim=-1)
        angle = 2.0 * torch.atan2(sin_half, w)
        angle = torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi

        axis = xyz / torch.clamp(sin_half.unsqueeze(-1), min=eps)
        default_axis = torch.zeros_like(axis)
        default_axis[:, 2] = 1.0
        small_mask = sin_half < eps
        axis[small_mask] = default_axis[small_mask]
        return axis, angle

    @classmethod
    def _axis_angle_to_quat(cls, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        axis = axis / torch.clamp(torch.norm(axis, dim=-1, keepdim=True), min=1e-8)

        half = 0.5 * angle
        s = torch.sin(half).unsqueeze(-1)
        c = torch.cos(half).unsqueeze(-1)
        q = torch.cat([axis * s, c], dim=-1)
        return cls._quat_normalize(q)

    @classmethod
    def _quat_step_toward(
        cls,
        current_q: torch.Tensor,
        target_q: torch.Tensor,
        max_angle_step_deg: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_q = cls._quat_normalize(current_q)
        target_q = cls._quat_normalize(target_q)

        q_err = cls._quat_mul(target_q, cls._quat_conjugate(current_q))
        q_err = cls._quat_normalize(q_err)

        axis, angle = cls._quat_to_axis_angle(q_err)
        max_angle_step = torch.deg2rad(
            torch.tensor(max_angle_step_deg, device=current_q.device, dtype=current_q.dtype)
        )
        step_angle = torch.clamp(angle, min=-max_angle_step, max=max_angle_step)

        q_step = cls._axis_angle_to_quat(axis, step_angle)
        return q_step, angle

    def _scripted_action(self, insert_meta_data: dict, pre_action_eef_internal_forces: float, isDisturb: bool) -> tuple[torch.Tensor, torch.Tensor]:
        current_pose = self._robot_ik_controller.eef_pose
        current_rot = current_pose[:3, :3]
        current_pos = current_pose[:3, 3]
        if self._teleop_hold_z is None:
            self._teleop_hold_z = float(current_pos[2])
        if isDisturb:
            aligned_pos = insert_meta_data["disturbed_pos"]
            aligned_rot = insert_meta_data["disturbed_rot"]
        else:
            aligned_pos = insert_meta_data["aligned_pos"]
            aligned_rot = insert_meta_data["aligned_rot"]
        delta_pos = aligned_pos - current_pos
        insert_action = self.compute_insert_action(torch.as_tensor(delta_pos, dtype=torch.float32)).numpy()
        # wrist_extrinsics = self._wrist_camera_extrinsics(np.asarray(current_pose, dtype=np.float32))
        # insert_action_cam_wrist = np.array([0., -0.005, 0.0], dtype=np.float32)
        # insert_action = self._wrist_camera_vector_to_world(insert_action_cam_wrist, wrist_extrinsics)
        target_pos = current_pos.copy()
        target_pos[:2] = target_pos[:2] + insert_action[:2] * 5
        # target_pos[2] += insert_action[2]
        if pre_action_eef_internal_forces > 0.5:
            insert_action[2] = -0.00002
        target_pos[2] = target_pos[2] + 0.0005 + insert_action[2] * 5
        # target_rot = aligned_rot.copy()
        target_rot, _, _, _ = self._interpolate_rotation_matrix(current_rot, aligned_rot)
        return target_pos, target_rot, insert_action
        


    def teleop_step(self, record_data=False, insert_meta_data: dict | None = None, episode_index: int | None = None, frame_index: int | None = None) -> tuple[dict, dict] | None:
        if not self.is_connected:
            raise RuntimeError("ScriptRobot is not connected. Run `robot.connect()` first.")

        isDisturb = episode_index is not None and frame_index is not None and episode_index < 8 and frame_index < 200
        isDisturb = False
        # Inference 
        inference = not self.config.debug
        before_fread_t = time.perf_counter()
        pre_action_eef_pose = np.asarray(self._robot_ik_controller.eef_pose, dtype=np.float32).copy()
        init_eef_pose = insert_meta_data["init_EEF_pose"]
        if inference:
            init_socket_depth_img = insert_meta_data["init_socket_depth_img"]
            init_plug_depth_img = np.asarray(insert_meta_data["init_plug_depth_img"], dtype=np.float32)
            aligned_socket_depth_img, _ = self._rotate_socket_depth_opposite_eef_yaw(
                init_socket_depth_img,
                init_eef_pose,
                pre_action_eef_pose,
            )
        pre_action_wrist_extrinsics = self._wrist_camera_extrinsics(pre_action_eef_pose)
        current_forces = self._get_eef_internal_forces(
                pre_action_eef_pose,
                pre_action_wrist_extrinsics,
            )
        if self.force_buffer is not None:
            pre_action_eef_internal_forces = current_forces - self.force_buffer
            print("Delta Force: ", pre_action_eef_internal_forces)
        else:
            pre_action_eef_internal_forces = torch.zeros(3, dtype=torch.float32)
            print("Delta Force: ", pre_action_eef_internal_forces)
        self.force_buffer = current_forces
        pre_action_wrist_images = None
        pre_action_wrist_depth = None
        if record_data and not isDisturb and "cam_wrist" in self.cameras:
            pre_action_wrist_images = read_zed_stereo_rgb(self.cameras["cam_wrist"])
            pre_action_wrist_depth = compute_foundation_stereo_depth(
                pre_action_wrist_images,
                self.cameras["cam_wrist"],
            )
            wrist_depth_input = downsample_depth_with_top_padding(pre_action_wrist_depth,target_h=360,target_w=640).copy()
            wrist_depth_input_tensor = torch.from_numpy(wrist_depth_input).unsqueeze(0).unsqueeze(0).float()
        
        self.logs["read_follower_dt_s"] = time.perf_counter() - before_fread_t

        current_rot_for_action = torch.as_tensor(
            pre_action_eef_pose[:3, :3],
            dtype=torch.float32,
        )
        # Inference 
        inference = not self.config.debug
        if inference:
            pre_action_eef_internal_forces_normalize = ((pre_action_eef_internal_forces - self.force_min) / (self.force_max - self.force_min + 1e-8)) * 2.0 -1.0
            force_input_tensor = torch.from_numpy(pre_action_eef_internal_forces_normalize.numpy()).unsqueeze(0).float()
            init_plug_photo_depth_tensor = torch.from_numpy(init_plug_depth_img).unsqueeze(0).float()
            socket_depth_tensor = torch.from_numpy(aligned_socket_depth_img).unsqueeze(0).float()
            # init_plug_photo_depth_tensor[...] = 0
            # socket_depth_tensor[...] = 0
            raw_actions = self.policy(depth = wrist_depth_input_tensor.cuda(), force = force_input_tensor.cuda(), init_plug_photo_depth= init_plug_photo_depth_tensor.cuda(), socket_depth = socket_depth_tensor.cuda())
            predict_actions=unnormalize_actions(raw_actions,self.pos_min,self.pos_max,self.rot_min,self.rot_max)[0].cpu()
        
        for i in range(5):
            if inference:
                init_eef_pose = insert_meta_data["init_EEF_pose"]
                action = predict_actions[i].cpu().numpy()
                current_pose = self._robot_ik_controller.eef_pose
                current_rot = current_pose[:3, :3]
                current_pos = current_pose[:3, 3]
                current_wrist_extrinsics = self._wrist_camera_extrinsics(np.asarray(current_pose, dtype=np.float32))
                action_pos_world = self._isaacgym_wrist_camera_vector_to_world(
                    action[:3],
                    current_wrist_extrinsics,
                )
                target_pos = current_pos.copy()
                target_pos[:2] = current_pos[:2] + action_pos_world[:2] * 5
                target_pos[2] = current_pos[2] + 0.0005 + action_pos_world[2] * 5
                # target_rot is rot6d -> rot mat @ current_rot
                action_rot6d = action[3:9]
                action_rot_mat = transforms.rotation_6d_to_matrix(torch.from_numpy(action_rot6d).float().unsqueeze(0)).squeeze(0).numpy()   
                target_rot_candidate = action_rot_mat @ current_rot
                target_rot = target_rot_candidate
                action9d = action.copy()
                # Debug Mode
                # if frame_index < 1:
                    # target_pos, target_rot, _ = self._scripted_action(insert_meta_data, pre_action_eef_internal_forces[2], isDisturb=False)
                # if frame_index > 15 and frame_index < 30:
                #     target_pos, _, _ = self._scripted_action(insert_meta_data, pre_action_eef_internal_forces[2], isDisturb=False)
                # if frame_index > 1:
                #     target_pos, target_rot, _ = self._scripted_action(insert_meta_data, pre_action_eef_internal_forces[2], isDisturb=False)

                # if frame_index > 30:
                #     target_pos, target_rot, _ = self._scripted_action(insert_meta_data, pre_action_eef_internal_forces[2], isDisturb=False)


            else:
                target_pos, target_rot, insert_action = self._scripted_action(insert_meta_data, pre_action_eef_internal_forces[2], isDisturb=isDisturb)
                target_rot_for_action = torch.as_tensor(target_rot, dtype=torch.float32)
                action_pos = torch.as_tensor(
                    self._world_vector_to_isaacgym_wrist_camera(insert_action, pre_action_wrist_extrinsics),
                    dtype=torch.float32,
                )
                relative_rot = target_rot_for_action @ current_rot_for_action.T
                relative_rot6d = transforms.matrix_to_rotation_6d(relative_rot[None]).squeeze(0)
                action9d = torch.cat([action_pos, relative_rot6d], dim=-1)

            before_fwrite_t = time.perf_counter()
            pybullet_control_success = self._control_refined(
                target_pos,
                target_rot,
                max_iters=self.config.step_refine_max_iters,
            )
            self.logs["write_follower_dt_s"] = time.perf_counter() - before_fwrite_t

            if not record_data or isDisturb:
                return

            images = {}
            for name in self.cameras:
                if name == "cam_wrist" and pre_action_wrist_images is not None:
                    continue
                before_camread_t = time.perf_counter()
                images[name] = self._camera_output_to_tensors(self.cameras[name].async_read())
                self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
                self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

            obs_dict, action_dict = {}, {}
            if inference:
                obs_dict["observation.init_plug_depth_img"] = torch.from_numpy(init_plug_depth_img)
                obs_dict["observation.aligned_socket_depth_img"] = torch.from_numpy(
                    np.asarray(aligned_socket_depth_img, dtype=np.float32)
                )
            obs_dict["observation.eef_internal_forces"] = pre_action_eef_internal_forces
            obs_dict["observation.eef_pose"] = pre_action_eef_pose
            obs_dict["observation.cam_wrist.extrinsics"] = pre_action_wrist_extrinsics
            action_dict["action"] = action9d
            for name in ["cam_wrist"]:
                if pre_action_wrist_images is not None:
                    image = torch.from_numpy(pre_action_wrist_images["left"])
                    obs_dict[f"observation.images.{name}.depth"] = np.asarray(pre_action_wrist_depth, dtype=np.float32)
                else:
                    image = images[name]["color"] if isinstance(images[name], dict) else images[name]
                obs_dict[f"observation.images.{name}"] = image
            for name in self.cameras:
                if name.startswith("cam_azure") and name in images:
                    image = images[name]["color"] if isinstance(images[name], dict) else images[name]
                    obs_dict[f"observation.images.{name}.color"] = image
        return obs_dict, action_dict
