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
from lerobot.common.utils.agos_canonical import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    PLUG_POINT_RADIUS,
    canonical_normalize,
    fingertip_points_to_world,
    paper_overlay,
    render_plug_canonical,
    render_world_canonical,
)
from lerobot.common.policy.agos_policy import AGOSPolicy


WRIST_CAM_TO_GRIPPER = np.array(
    [
        [-0.00768086, -0.94557934, -0.32530096, 0.07294499],
        [0.99995759, -0.00891583, 0.00230583, -0.03177615],
        [-0.00508067, -0.32526946, 0.94560772, -0.08727812],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
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
        # World-frame pose correction (command - target) that compensates gravity sag; warm-started
        # by _control_refined and updated once per step by _control_feedforward.
        self._ff_pos = np.zeros(3, dtype=np.float64)
        self._ff_rotvec = np.zeros(3, dtype=np.float64)
        # AGOS visuo-tactile diffusion policy (third_party/AGOS, e.g. agos_canonical_split_unet), built with
        # the AGOS code from the checkpoint's own config, EMA weights, normalizers and action frame.
        self.policy = None
        if self.config.agos_policy_ckpt:
            self.policy = AGOSPolicy(
                self.config.agos_policy_ckpt,
                device="cuda" if torch.cuda.is_available() else "cpu",
                rot_cap_deg=self.config.teleop_max_rotation_step_deg,
                rot_deadband_deg=self.config.teleop_rot_deadband_deg,
            )
        self._agos_force_bias = None

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
            "observation.plug_canonical_depth": {
                "dtype": "float32",
                "shape": (CANONICAL_HEIGHT, CANONICAL_WIDTH),
                "names": ["height", "width"],
                "info": "AGOS plug canonical depth (m, NaN = empty): fingertip-frame plug cloud moved "
                "with the current fingertip pose, world axes, bottom-up, own xy-mean centre",
            },
            "observation.plug_canonical_center_xy": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["x", "y"],
            },
            "observation.socket_canonical_depth": {
                "dtype": "float32",
                "shape": (CANONICAL_HEIGHT, CANONICAL_WIDTH),
                "names": ["height", "width"],
                "info": "AGOS socket canonical depth (m, NaN = empty): fused wrist scan, world axes, "
                "top-down, bbox centre; constant within an episode",
            },
            "observation.points.init_plug_points_fingertip": {
                "dtype": "pcd",
                "shape": (-1, 3),
                "names": ["points", "xyz"],
            },
            "observation.points.initial_wrist_points_world": {
                "dtype": "pcd",
                "shape": (-1, 3),
                "names": ["points", "xyz"],
            },
        }
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


    def _control_refined(self, target_pos, target_rot, max_iters: int | None = None, verbose: bool = True) -> bool:
        """Move to the full 6-DoF target pose with closed-loop refinement (see RobotIKController.control_refined)."""
        cfg = self.config
        if not cfg.gravity_compensation:
            return self._robot_ik_controller.control(
                target_pos=target_pos,
                target_rot=target_rot,
                grasping_action=getattr(self, "_last_gripper_action", cfg.gripper_open_action),
                wait_times=50,
                joint_threshold=float(cfg.script_joint_solution_threshold),
            )
        ok = self._robot_ik_controller.control_refined(
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
        # Warm-start the per-step feedforward with the correction that worked here.
        self._ff_pos = np.asarray(self._robot_ik_controller.last_pos_correction, dtype=np.float64).copy()
        self._ff_rotvec = np.asarray(self._robot_ik_controller.last_rot_correction, dtype=np.float64).copy()
        return ok

    def _control_feedforward(self, target_pos, target_rot) -> bool:
        """One control call per step with gravity compensation carried across steps.

        Commands target + feedforward correction, then updates the correction from the
        measured error (a low-gain integral across steps). The sag varies slowly with the
        arm configuration, so this tracks it without extra iterations or settle ticks.
        """
        cfg = self.config
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_r = R.from_matrix(np.asarray(target_rot, dtype=np.float64).reshape(3, 3))
        ok = self._robot_ik_controller.control(
            target_pos=target_pos + self._ff_pos,
            target_rot=(R.from_rotvec(self._ff_rotvec) * target_r).as_matrix(),
            grasping_action=getattr(self, "_last_gripper_action", cfg.gripper_open_action),
            wait_times=50,
            joint_threshold=float(cfg.script_joint_solution_threshold),
        )
        pose = np.asarray(self._robot_ik_controller.eef_pose, dtype=np.float64)
        pos_err = target_pos - pose[:3, 3]
        rot_err = (target_r * R.from_matrix(pose[:3, :3]).inv()).as_rotvec()
        print(
            f"[script] step pose error (gravity_compensation={cfg.gravity_compensation}): "
            f"pos={np.linalg.norm(pos_err) * 1e3:.2f} mm, rot={np.rad2deg(np.linalg.norm(rot_err)):.3f} deg"
        )
        gain = float(cfg.step_feedforward_gain)
        if cfg.gravity_compensation and gain > 0:
            self._ff_pos = self._ff_pos + gain * pos_err
            self._ff_rotvec = (R.from_rotvec(gain * rot_err) * R.from_rotvec(self._ff_rotvec)).as_rotvec()
            pos_norm = np.linalg.norm(self._ff_pos)
            if pos_norm > cfg.pose_refine_max_pos_correction:
                self._ff_pos *= cfg.pose_refine_max_pos_correction / pos_norm
            rot_norm = np.linalg.norm(self._ff_rotvec)
            max_rot = np.deg2rad(cfg.pose_refine_max_rot_correction_deg)
            if rot_norm > max_rot:
                self._ff_rotvec *= max_rot / rot_norm
        return ok

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

    def _canonical_observation(self, insert_meta_data: dict, fingertip_pose: np.ndarray):
        """AGOS canonical_observation(): (plug depth, plug centre xy, socket depth), raw metres, NaN = empty.

        The plug cloud stored in the fingertip frame is moved with the current fingertip pose
        (proprioception) and rendered bottom-up around its own xy mean; the socket view is the
        one-off render of the fused wrist scan. A missing scan gives an all-NaN image, like a
        failed plug photo in sim.
        """
        empty = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH), np.nan, dtype=np.float32)
        plug_points = insert_meta_data.get("init_plug_points_fingertip")
        if plug_points is not None and len(plug_points) > 0:
            plug_view = render_plug_canonical(plug_points, insert_meta_data["init_plug_colors"], fingertip_pose)
            plug_depth = plug_view["depth"].astype(np.float32)
            plug_center = np.asarray(plug_view["center_xy"], dtype=np.float32)
        else:
            plug_depth, plug_center = empty.copy(), np.full(2, np.nan, dtype=np.float32)
        socket_depth = insert_meta_data.get("socket_canonical_depth")
        socket_depth = empty.copy() if socket_depth is None else np.asarray(socket_depth, dtype=np.float32)
        if (plug_points is None or socket_depth is None or not np.isfinite(socket_depth).any()) and not getattr(
            self, "_warned_missing_canonical", False
        ):
            print("[script] Plug or socket scan missing; recording NaN canonical images for it.")
            self._warned_missing_canonical = True
        return plug_depth, plug_center, socket_depth

    def _record_agos_visualization(self, insert_meta_data: dict, fingertip_pose: np.ndarray, wrist_rgb) -> None:
        """Store one (wrist RGB, AGOS overlay) pair.

        The overlay is G_t = {plug virtual image I_p^t, socket virtual image I_s^0}: the two policy
        inputs, each rendered about its own centre (plug: xy mean, socket: bbox), drawn on top of
        each other.
        """
        if not self.config.agos_visualize or (self._agos_vis_frames_count - 1) % max(1, self.config.agos_vis_every):
            return
        plug_points = insert_meta_data.get("init_plug_points_fingertip")
        socket_view = insert_meta_data.get("socket_canonical_view")
        if plug_points is None or len(plug_points) == 0:
            return
        if socket_view is None:
            empty = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH), np.nan, dtype=np.float32)
            socket_view = {"mask": np.zeros_like(empty, dtype=bool), "depth": empty, "center_xy": None}
        plug_world = fingertip_points_to_world(plug_points, fingertip_pose)
        plug_view = render_world_canonical(
            plug_world, insert_meta_data["init_plug_colors"], +1, point_radius=PLUG_POINT_RADIUS, center_mode="mean"
        )
        if wrist_rgb is not None:
            wrist_rgb = np.asarray(wrist_rgb)[::4, ::4].copy()  # 720x1280 -> 180x320
        frames = getattr(self, "_agos_vis_frames", None)
        if frames is None:
            frames = self._agos_vis_frames = []
        frames.append({"wrist": wrist_rgb, "overlay": paper_overlay(plug_view, socket_view)})

    def on_manual_pause(self) -> None:
        """Space pressed: stop commanding so the arm can be hand-guided (Franka guiding button)."""
        self._pause_pose = np.asarray(self._robot_ik_controller.eef_pose, dtype=np.float64).copy()
        print("[script] PAUSED: no commands are sent. Hand-guide the arm, release the guiding button, "
              "then press Space to resume.")

    def on_manual_resume(self, fresh_state_timeout_s: float = 2.0) -> None:
        """Re-sync after a manual move: wait for a fresh robot state, drop the gravity feedforward
        (it would treat the hand push as tracking error and drive the arm back), continue from the
        measured pose."""
        n0 = self.robot_interface.state_buffer_size
        t0 = time.perf_counter()
        while self.robot_interface.state_buffer_size <= n0 + 2 and time.perf_counter() - t0 < fresh_state_timeout_s:
            time.sleep(0.01)
        if self.robot_interface.state_buffer_size <= n0:
            print("[script] WARNING: no fresh robot state after resume; the pose may be stale.")
        self._ff_pos[:] = 0.0
        self._ff_rotvec[:] = 0.0
        self._teleop_hold_z = None
        pose = np.asarray(self._robot_ik_controller.eef_pose, dtype=np.float64)
        moved_mm = np.linalg.norm(pose[:3, 3] - self._pause_pose[:3, 3]) * 1e3 if getattr(self, "_pause_pose", None) is not None else float("nan")
        moved_deg = np.rad2deg((R.from_matrix(pose[:3, :3]) * R.from_matrix(self._pause_pose[:3, :3]).inv()).magnitude()) \
            if getattr(self, "_pause_pose", None) is not None else float("nan")
        print(f"[script] RESUMED from the measured pose (moved {moved_mm:.1f} mm, {moved_deg:.1f} deg while paused); "
              "feedforward reset.")

    def _socket_force_world(self, frame_index) -> np.ndarray:
        """AGOS force input: socket contact force in the world frame (N).

        Sim uses the net contact force on the socket. On the robot this is the reaction of the
        Franka external-force estimate at the EE (O_F_ext_hat_K, base frame): sign * (F_ext - bias),
        with the bias taken in free space at the first step of each episode (plug weight, model error).
        """
        f = np.zeros(3, dtype=np.float64)
        if self.robot_interface is not None and self.robot_interface.state_buffer_size > 0:
            state = self.robot_interface._state_buffer[-1]
            if hasattr(state, "O_F_ext_hat_K"):
                f = np.asarray(state.O_F_ext_hat_K, dtype=np.float64).reshape(-1)[:3]
        if self._agos_force_bias is None or frame_index == 0:
            self._agos_force_bias = f.copy()
        return (float(self.config.agos_force_sign) * (f - self._agos_force_bias)).astype(np.float32)

    def _cap_translation_step(self, delta: np.ndarray) -> np.ndarray:
        """Clip a per-action translation to teleop_max_translation_step_m (AGOS max_translation_step)."""
        delta = np.asarray(delta, dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        cap = float(self.config.teleop_max_translation_step_m)
        return delta * (cap / norm) if cap > 0 and norm > cap else delta

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
        cfg = self.config
        raw_step = np.zeros(3, dtype=np.float64)
        raw_step[:2] = insert_action[:2]
        # target_pos[2] += insert_action[2]
        if pre_action_eef_internal_forces > 0.5:
            insert_action[2] = -0.00002
        raw_step[2] = cfg.teleop_z_bias_m + insert_action[2]
        raw_step = self._cap_translation_step(raw_step)  # AGOS-scale step (<= 0.3 mm)
        target_pos = current_pos + raw_step * cfg.teleop_translation_gain  # amplified for execution
        # target_rot = aligned_rot.copy()
        target_rot, _, _, _ = self._interpolate_rotation_matrix(
            current_rot, aligned_rot, max_angle_step_deg=cfg.teleop_max_rotation_step_deg
        )
        # Record the AGOS-scale step (before the execution gain), matching the policy's action scale.
        return target_pos, target_rot, np.asarray(raw_step, dtype=np.float32)
        


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
        plug_canonical_depth, plug_canonical_center_xy, socket_canonical_depth = self._canonical_observation(
            insert_meta_data, pre_action_eef_pose
        )
        self._agos_vis_frames_count = getattr(self, "_agos_vis_frames_count", 0) + 1
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
            # 720 x 1280 -> 360 x 640 by sampling every 2nd pixel (checkpoint depth resolution).
            wrist_depth_input = np.asarray(pre_action_wrist_depth, dtype=np.float32)[::2, ::2].copy()
        
        self.logs["read_follower_dt_s"] = time.perf_counter() - before_fread_t

        current_rot_for_action = torch.as_tensor(
            pre_action_eef_pose[:3, :3],
            dtype=torch.float32,
        )
        # Inference 
        self._record_agos_visualization(
            insert_meta_data,
            pre_action_eef_pose,
            pre_action_wrist_images["left"] if pre_action_wrist_images is not None else None,
        )
        inference = not self.config.debug
        # Socket contact force (world), AGOS force input: measured before this prediction, i.e. after
        # the previous action (training time_offset = -1).
        socket_force_world = self._socket_force_world(frame_index)
        predict_actions = None
        run_policy = self.policy is not None and (inference or self.config.agos_shadow_predict)
        if run_policy and pre_action_wrist_depth is not None:
            obs = self.policy.build_obs(
                depth_m=wrist_depth_input,
                plug_depth=plug_canonical_depth,
                socket_depth=socket_canonical_depth,
                socket_force_world=socket_force_world,
            )
            predict_actions = self.policy.predict(obs)  # [10, 9] training frame, rotation-limited
        elif inference:
            raise RuntimeError("Policy inference needs the AGOS checkpoint (agos_policy_ckpt) and the wrist depth.")
        
        for i in range(5):
            if inference:
                # AGOS execution: map the step from the training frame (eef) to world with the *live*
                # fingertip pose, then target = current + dp_w, R_target = R_w @ R_current.
                action = predict_actions[i]
                current_pose = np.asarray(self._robot_ik_controller.eef_pose, dtype=np.float64)
                current_rot = current_pose[:3, :3]
                current_pos = current_pose[:3, 3]
                world_action = self.policy.action_to_world(action, R.from_matrix(current_rot).as_quat())
                target_pos = current_pos + self._cap_translation_step(world_action[:3]) * self.config.teleop_translation_gain
                rot_world = transforms.rotation_6d_to_matrix(torch.as_tensor(world_action[3:9])[None])[0].numpy()
                target_rot = rot_world @ current_rot
                action9d = action.numpy().copy()
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
                if i == 0 and predict_actions is not None:
                    # Shadow mode: what the AGOS policy would do now vs the scripted step (world frame).
                    cur = np.asarray(self._robot_ik_controller.eef_pose, dtype=np.float64)
                    w = self.policy.action_to_world(predict_actions[0], R.from_matrix(cur[:3, :3]).as_quat())
                    pol_rot = np.rad2deg(R.from_matrix(
                        transforms.rotation_6d_to_matrix(torch.as_tensor(w[3:9])[None])[0].numpy()).magnitude())
                    scr_rot = np.rad2deg((R.from_matrix(target_rot) * R.from_matrix(cur[:3, :3]).inv()).magnitude())
                    print(f"[agos shadow] policy dpos {np.round(w[:3] * 1e3, 3).tolist()} mm, rot {pol_rot:.2f} deg | "
                          f"scripted dpos {np.round((target_pos - cur[:3, 3]) * 1e3, 3).tolist()} mm, rot {scr_rot:.2f} deg | "
                          f"socket force {np.round(socket_force_world, 2).tolist()} N")
                target_rot_for_action = torch.as_tensor(target_rot, dtype=torch.float32)
                action_pos = torch.as_tensor(
                    self._world_vector_to_isaacgym_wrist_camera(insert_action, pre_action_wrist_extrinsics),
                    dtype=torch.float32,
                )
                relative_rot = target_rot_for_action @ current_rot_for_action.T
                relative_rot6d = transforms.matrix_to_rotation_6d(relative_rot[None]).squeeze(0)
                action9d = torch.cat([action_pos, relative_rot6d], dim=-1)

            before_fwrite_t = time.perf_counter()
            pose_before = np.asarray(self._robot_ik_controller.eef_pose, dtype=np.float64).copy()
            pybullet_control_success = self._control_feedforward(target_pos, target_rot)
            # Pace actions like AGOS (10 Hz: 6 sim steps of 1/60 s): 0.5 deg/action is 5 deg/s only at
            # that rate; without it the next action starts as soon as the joints converge.
            min_period = float(self.config.teleop_min_action_period_s)
            remaining = min_period - (time.perf_counter() - before_fwrite_t)
            if remaining > 0:
                time.sleep(remaining)
            action_dt = time.perf_counter() - before_fwrite_t
            self.logs["write_follower_dt_s"] = action_dt
            if i == 0:
                pose_after = np.asarray(self._robot_ik_controller.eef_pose, dtype=np.float64)
                rot_deg = np.rad2deg((R.from_matrix(pose_after[:3, :3]) * R.from_matrix(pose_before[:3, :3]).inv()).magnitude())
                trans_mm = np.linalg.norm(pose_after[:3, 3] - pose_before[:3, 3]) * 1e3
                print(f"[script] action pace: {1.0 / max(action_dt, 1e-6):.1f} actions/s, measured "
                      f"{rot_deg / max(action_dt, 1e-6):.1f} deg/s, {trans_mm / max(action_dt, 1e-6):.1f} mm/s")

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
            obs_dict["observation.plug_canonical_depth"] = plug_canonical_depth
            obs_dict["observation.plug_canonical_center_xy"] = plug_canonical_center_xy
            obs_dict["observation.socket_canonical_depth"] = socket_canonical_depth
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
