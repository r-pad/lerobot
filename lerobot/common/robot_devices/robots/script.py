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
    depth_meters_to_uint16_mm,
    droid_ik_model_eef_pose,
    get_zed_intrinsics_and_baseline,
    print_droid_ik_diagnostics,
    read_zed_stereo_rgb,
)
from lerobot.common.robot_devices.robots.configs import ScriptRobotConfig
from lerobot.common.robot_devices.robots.droid import DroidRobot
from lerobot.common.robot_devices.robots.robot_ik_controller import RobotIKController


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

    @property
    def camera_features(self) -> dict:
        if "cam_wrist" not in self.cameras:
            return super().camera_features

        cam_cfg = self.cameras["cam_wrist"].config
        return {
            "observation.images.cam_wrist": {
                "shape": (cam_cfg.height, cam_cfg.width, cam_cfg.channels),
                "names": ["height", "width", "channels"],
                "info": f"{cam_cfg.color_mode.upper()} color image",
            },
            "observation.images.cam_wrist.depth": {
                "shape": (cam_cfg.height, cam_cfg.width, 1),
                "names": ["height", "width", "channels"],
                "info": "FoundationStereo depth from wrist ZED Mini in uint16 millimeters",
            }
        }

    @property
    def motor_features(self) -> dict:
        return {
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
                "shape": (6,),
                "names": ["fx", "fy", "fz", "tx", "ty", "tz"],
            },
            "observation.eef_pose": {
                "dtype": "float32",
                "shape": (4, 4),
                "names": ["row", "col"],
            },
        }
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
    def _get_eef_internal_forces(self) -> torch.Tensor:
        if self.robot_interface.state_buffer_size == 0:
            return torch.zeros(6, dtype=torch.float32)

        state = self.robot_interface._state_buffer[-1]
        for attr in ("K_F_ext_hat_K", "O_F_ext_hat_K"):
            if hasattr(state, attr):
                wrench = np.asarray(getattr(state, attr), dtype=np.float32).reshape(-1)
                if wrench.size >= 6:
                    return torch.from_numpy(wrench[:6].copy())

        return torch.zeros(6, dtype=torch.float32)

    def connect(self):
        if self.is_connected:
            raise RuntimeError("ScriptRobot is already connected. Do not run `robot.connect()` twice.")

        from deoxys.franka_interface import FrankaInterface
        from deoxys.utils import YamlConfig

        self.robot_interface = FrankaInterface(
            self.config.deoxys_general_cfg_file,
            use_visualizer=False,
            has_gripper=False,
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

        action[2] = 0.0
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

    def _scripted_action(self, insert_meta_data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        current_pose = self._robot_ik_controller.eef_pose
        current_rot = current_pose[:3, :3]
        current_pos = current_pose[:3, 3]
        if self._teleop_hold_z is None:
            self._teleop_hold_z = float(current_pos[2])
        aligned_pos = insert_meta_data["aligned_pos"]
        aligned_rot = insert_meta_data["aligned_rot"]
        delta_pos = aligned_pos - current_pos
        insert_action = self.compute_insert_action(torch.as_tensor(delta_pos, dtype=torch.float32)).numpy()
        target_pos = current_pos.copy()
        target_pos[:2] = target_pos[:2] + insert_action[:2] * 5
        target_pos[2] = self._teleop_hold_z + 0.0005 # This is for compensating gravity
        target_rot, _, _, _ = self._interpolate_rotation_matrix(current_rot, aligned_rot)
        return target_pos, target_rot, insert_action
        


    def teleop_step(self, record_data=False, insert_meta_data: dict | None = None) -> tuple[dict, dict] | None:
        if not self.is_connected:
            raise RuntimeError("ScriptRobot is not connected. Run `robot.connect()` first.")
        
        before_fread_t = time.perf_counter()
        pre_action_eef_pose = np.asarray(self._robot_ik_controller.eef_pose, dtype=np.float32).copy()
        pre_action_eef_internal_forces = self._get_eef_internal_forces()
        pre_action_wrist_images = None
        pre_action_wrist_depth = None
        if record_data and "cam_wrist" in self.cameras:
            pre_action_wrist_images = read_zed_stereo_rgb(self.cameras["cam_wrist"])
            pre_action_wrist_depth = compute_foundation_stereo_depth(
                pre_action_wrist_images,
                self.cameras["cam_wrist"],
            )
            # if not getattr(self, "_debug_first_wrist_point_cloud_done", False):
            #     k, _ = get_zed_intrinsics_and_baseline(self.cameras["cam_wrist"])
            #     _visualize_depth_point_cloud_once(
            #         pre_action_wrist_depth,
            #         pre_action_wrist_images["left"],
            #         k,
            #     )
            #     self._debug_first_wrist_point_cloud_done = True
                # import pdb; pdb.set_trace()
        self.logs["read_follower_dt_s"] = time.perf_counter() - before_fread_t

        current_rot_for_action = torch.as_tensor(
            pre_action_eef_pose[:3, :3],
            dtype=torch.float32,
        )
        target_pos, target_rot, insert_action = self._scripted_action(insert_meta_data)
        target_rot_for_action = torch.as_tensor(target_rot, dtype=torch.float32)
        action_pos = torch.as_tensor(insert_action, dtype=torch.float32)
        relative_rot = target_rot_for_action @ current_rot_for_action.T
        relative_rot6d = transforms.matrix_to_rotation_6d(relative_rot[None]).squeeze(0)
        action9d = torch.cat([action_pos, relative_rot6d], dim=-1)

        before_fwrite_t = time.perf_counter()
        pybullet_control_success = self._robot_ik_controller.control(
            target_pos=target_pos,
            target_rot=target_rot,
            grasping_action=getattr(self, "_last_gripper_action", self.config.gripper_open_action),
            wait_times=50,
            joint_threshold=float(getattr(self.config, "script_joint_solution_threshold", 0.5)),
            )
        self.logs["write_follower_dt_s"] = time.perf_counter() - before_fwrite_t

        if not record_data:
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
        obs_dict["observation.eef_internal_forces"] = pre_action_eef_internal_forces
        obs_dict["observation.eef_pose"] = pre_action_eef_pose
        action_dict["action"] = action9d
        for name in ["cam_wrist"]:
            if pre_action_wrist_images is not None:
                image = torch.from_numpy(pre_action_wrist_images["left"])
                obs_dict[f"observation.images.{name}.depth"] = depth_meters_to_uint16_mm(pre_action_wrist_depth)
            else:
                image = images[name]["color"] if isinstance(images[name], dict) else images[name]
            obs_dict[f"observation.images.{name}"] = image
        return obs_dict, action_dict
