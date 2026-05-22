"""ScriptRobot: direct scripted Franka motion without a GELLO leader."""

import json
import os
import time

import numpy as np
import torch
import pytorch3d.transforms as transforms
from scipy.spatial.transform import Rotation as R

from lerobot.common.robot_devices.control_utils import (
    droid_ik_model_eef_pose,
    print_droid_ik_diagnostics,
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

    @property
    def motor_features(self) -> dict:
        return super().motor_features

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

        self._init_robotiq_gripper_without_gello()
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

    def _scripted_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current_rot, current_pos = self._pybullet_current_eef_pose(state[:7].detach().cpu().numpy())
        if self._pose_target_pos is None:
            self._pose_target_pos = current_pos.copy()
            self._pose_target_origin_pos = current_pos.copy()
            self._pose_target_rot = current_rot.copy()

        delta_pos = self._script_delta_pos(state)
        if self._last_joint_target_reached:
            self._last_script_delta_pos = delta_pos.copy()
            self._pose_target_pos = self._pose_target_pos + delta_pos
        else:
            self._last_script_delta_pos = np.zeros(3, dtype=np.float64)
        target_pos = self._pose_target_pos.copy()
        target_rot = self._pose_target_rot if self._pose_target_rot is not None else current_rot
        target_rot_6d = transforms.matrix_to_rotation_6d(
            torch.from_numpy(target_rot[None])
        ).squeeze()
        target_eef = torch.cat(
            [target_rot_6d, torch.from_numpy(target_pos), state[-1:]],
            dim=0,
        ).float()

        action = state.clone()
        action[-1] = self.config.gripper_close_action
        return action, target_eef


    def teleop_step(self, record_data=False, insert_meta_data: dict | None = None) -> tuple[dict, dict] | None:
        if not self.is_connected:
            raise RuntimeError("ScriptRobot is not connected. Run `robot.connect()` first.")
        
        before_fread_t = time.perf_counter()
        franka_joints = self._get_franka_joints()
        gripper_width = self._get_gripper_width()
        self.logs["read_follower_dt_s"] = time.perf_counter() - before_fread_t

        state = torch.tensor(list(franka_joints) + [gripper_width], dtype=torch.float32)
        last_joint_gap_before_tick = None
        if self._last_joint_target is not None:
            last_joint_gap_before_tick = float(np.max(np.abs(franka_joints - self._last_joint_target)))
            tolerance = float(getattr(self.config, "script_joint_convergence_tolerance", 1e-3))
            self._last_joint_target_reached = last_joint_gap_before_tick < tolerance
        else:
            self._last_joint_target_reached = True
        action_tensor, target_eef = self._scripted_action(state)
        should_log_pose = self._script_step_count % 30 == 0
        before_ik_pos = None
        target_ik_pos = None
        target_rot = transforms.rotation_6d_to_matrix(target_eef[:6][None]).squeeze().numpy()
        target_pos = target_eef[6:9].numpy()
        if should_log_pose:
            _, before_ik_pos = self._pybullet_current_eef_pose(state[:7].detach().cpu().numpy())
            target_ik_pos = target_pos
        self._script_step_count += 1

        before_fwrite_t = time.perf_counter()
        control_path = "robot_ik_unavailable"
        if self._robot_ik_controller is not None:
            control_path = "pybullet_robot_ik_controller"
            if self.config.gripper_close_action < self.config.gripper_threshold:
                grasping_action = self.config.gripper_close_action
            else:
                grasping_action = self.config.gripper_open_action
            if not hasattr(self, "_last_gripper_action") or grasping_action != self._last_gripper_action:
                if grasping_action == self.config.gripper_close_action:
                    self.robotiq_gripper.close()
                else:
                    self.robotiq_gripper.open()
                self._last_gripper_action = grasping_action

            pybullet_control_success = self._robot_ik_controller.control(
                target_pos=target_pos,
                target_rot=target_rot,
                grasping_action=grasping_action,
                # wait_times=int(getattr(self.config, "script_joint_wait_times", 50)),
                wait_times = 500,
                joint_threshold=float(getattr(self.config, "script_joint_solution_threshold", 0.5)),
            )
            if self._robot_ik_controller.last_action is not None:
                action_tensor = torch.as_tensor(self._robot_ik_controller.last_action, dtype=torch.float32)
            if self._robot_ik_controller.last_joint_target is not None:
                self._last_joint_target = np.asarray(
                    self._robot_ik_controller.last_joint_target,
                    dtype=np.float64,
                )
        else:
            raise RuntimeError("RobotIKController is required; refusing to use MuJoCo fallback.")
        self.logs["write_follower_dt_s"] = time.perf_counter() - before_fwrite_t
        if should_log_pose:
            time.sleep(0.03)
            after_joints = self._get_franka_joints()
            _, after_ik_pos = self._pybullet_current_eef_pose(after_joints)
            step_delta = self._last_script_delta_pos
            accumulated_target_delta = target_ik_pos - self._pose_target_origin_pos
            tracking_gap_before = target_ik_pos - before_ik_pos
            actual_motion = after_ik_pos - before_ik_pos
            tracking_error_after = target_ik_pos - after_ik_pos
            commanded_joints = action_tensor[:7].detach().cpu().numpy()
            measured_joints_before = state[:7].detach().cpu().numpy()
            joint_command_delta = commanded_joints - measured_joints_before
            joint_motion_after = after_joints - measured_joints_before
            controller_debug = (
                "\n"
                f"control_path={control_path} "
                f"target_advanced_this_tick={bool(np.any(step_delta))} "
                f"last_joint_target_reached_before_tick={self._last_joint_target_reached} "
                f"last_joint_gap_before_tick={last_joint_gap_before_tick} "
                f"joint_command_delta_norm={float(np.linalg.norm(joint_command_delta)):.6f} "
                f"joint_motion_after_norm={float(np.linalg.norm(joint_motion_after)):.6f} \n"
                f"commanded_joints={[round(v, 6) for v in commanded_joints.tolist()]} \n"
                f"joint_command_delta={[round(v, 6) for v in joint_command_delta.tolist()]} \n"
                f"joint_motion_after={[round(v, 6) for v in joint_motion_after.tolist()]}"
            )
            if self._robot_ik_controller is not None:
                controller_debug += (
                    " \n"
                    f"pybullet_control_success={getattr(self._robot_ik_controller, 'last_control_success', None)} "
                    f"pybullet_joint_target={np.round(getattr(self._robot_ik_controller, 'last_joint_target', []), 6).tolist()} "
                    f"pybullet_action={np.round(getattr(self._robot_ik_controller, 'last_action', []), 6).tolist()}"
                )
            print(
                "[script:pose] \n"
                f"step_delta_command_this_tick={[round(v, 6) for v in step_delta.tolist()]} \n"
                f"accumulated_target_delta_from_start={[round(v, 6) for v in accumulated_target_delta.tolist()]} \n"
                f"before_current_pos={[round(v, 4) for v in before_ik_pos.tolist()]} \n"
                f"target_accumulated_pos={[round(v, 4) for v in target_ik_pos.tolist()]} \n"
                f"after_current_pos={[round(v, 4) for v in after_ik_pos.tolist()]} \n"
                f"tracking_gap_before_send={[round(v, 6) for v in tracking_gap_before.tolist()]} \n"
                f"actual_motion_after_send={[round(v, 6) for v in actual_motion.tolist()]} \n"
                f"tracking_error_after_send={[round(v, 6) for v in tracking_error_after.tolist()]}"
                f"{controller_debug}"
            )
            print()

        if not record_data:
            return

        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self._camera_output_to_tensors(self.cameras[name].async_read())
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action_tensor
        for name in self.cameras:
            if type(images[name]) == dict:
                for img_name in images[name].keys():
                    obs_dict[f"observation.images.{name}.{img_name}"] = images[name][img_name]
            else:
                obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict
