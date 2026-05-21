"""ScriptRobot: direct scripted Franka motion without a GELLO leader."""

import time

import numpy as np
import torch
import pytorch3d.transforms as transforms

from deoxys.utils import transform_utils

from lerobot.common.policies.robot_adapters import DroidAdapter
from lerobot.common.robot_devices.robots.configs import ScriptRobotConfig
from lerobot.common.robot_devices.robots.droid import DroidRobot
from lerobot.common.robot_devices.robots.robot_controller import FrankaOSCController


class ScriptRobot(DroidRobot):
    """Droid-like Franka robot that executes a built-in scripted motion.

    Each control step lifts the current end-effector target along world Z by
    ``config.z_step`` metres, converts that EEF target to joint space, and sends
    it through the same deoxys joint controller used by DroidRobot.
    """

    robot_type = "script"

    def __init__(self, config: ScriptRobotConfig | None = None, **kwargs):
        super().__init__(config if config is not None else ScriptRobotConfig(**kwargs))
        self._adapter = None
        self._script_step_count = 0
        self._script_joint_target = None
        self._home_move_done = False
        self._pose_move_done = False
        self._pose_controller = None
        self._pose_target_pos = None

    @property
    def motor_features(self) -> dict:
        motor_features = super().motor_features
        motor_features["action"] = {
            "dtype": "float32",
            "shape": (8,),
            "names": ["target_x", "target_y", "target_z", "target_qx", "target_qy", "target_qz", "target_qw", "gripper"],
        }
        return motor_features

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
        self._adapter = DroidAdapter(action_space="right_eef")

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
        self._pose_controller = FrankaOSCController(
            controller_type=self.config.deoxys_controller_type,
            controller_cfg=self.config.deoxys_controller_cfg_file,
            robot_interface=self.robot_interface,
            tip_offset=np.zeros(3),
            verbose=False,
            pos_action_gain=self.config.pose_pos_action_gain,
            rot_action_gain=self.config.pose_rot_action_gain,
        )

        self.is_connected = True
        print(
            "[ScriptRobot] Connected. "
            f"script_mode={self.config.script_mode} controller={self.config.deoxys_controller_type}"
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
        print("Current EEF pose (pos + rot_6d): "
              f"{[round(v, 4) for v in  trans.tolist()]} + "
              f"{[round(v, 4) for v in rot_6d.tolist()]}"
              )        
        return torch.cat([rot_6d, trans, state[-1:]], axis=0).float()

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

    def _scripted_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        current_pose = self._pose_controller.eef_pose
        current_pos = current_pose[:3, 3]
        original_pos = current_pos.copy()
        current_rot_6d = transforms.matrix_to_rotation_6d(
            torch.from_numpy(current_pose[:3, :3][None])
        ).squeeze()
        current_rot = transforms.rotation_6d_to_matrix(current_rot_6d[None]).squeeze().numpy()
        target_quat = transform_utils.mat2quat(current_rot)
        self._pose_target_pos = current_pos + np.array([0.0000, 0.0000, 0.0005])
        target_pos = self._pose_target_pos.copy()
        target_eef = torch.cat(
            [current_rot_6d, torch.from_numpy(target_pos), state[-1:]],
            axis=0,
        ).float()
        
        gripper_action = self.config.gripper_close_action

        if self._pose_controller is None:
            raise RuntimeError("Pose controller is not initialized. Run robot.connect() first.")
        # pos_diff, angle_diff = self._pose_controller.move_to(
        #     target_pos=target_pos,
        #     target_quat=target_quat,
        #     grasp=None,
        #     num_steps=num_steps,
        #     num_additional_steps=num_addition_steps,
        #     pos_tolerance=self.config.pose_pos_tolerance,
        #     rot_tolerance=self.config.pose_rot_tolerance,
        #     max_delta_pos=self.config.pose_max_delta_pos,
        #     action_smoothing=self.config.pose_action_smoothing,
        # )
        if self._script_step_count % 30 == 0:
            raw_flange_pose = self.robot_interface.last_eef_pose
            raw_flange_pos = raw_flange_pose[:3, 3]
            tip_pose = self._pose_controller.eef_pose
            tip_pos = tip_pose[:3, 3]
            target_flange_pos, target_flange_quat = self._pose_controller.target_to_flange_pose(
                target_pos, target_quat
            )
            print(
                "[script:pose] target_pos="
                f"{np.round(target_pos, 4).tolist()} current_pos={np.round(tip_pos, 4).tolist()} "
                f"pos_error={np.linalg.norm(target_pos - tip_pos):.4f} m"
            )
            print(
                "[script:pose] target_flange_pos="
                f"{np.round(target_flange_pos.squeeze(), 4).tolist()} "
                f"current_flange_pos={np.round(raw_flange_pos, 4).tolist()} "
                f"flange_error={np.linalg.norm(target_flange_pos.squeeze() - raw_flange_pos):.4f} m "
                f"target_flange_quat={np.round(target_flange_quat, 4).tolist()}"
            )
            print(
                "[script:pose] target_eef="
                f"{[round(v, 4) for v in target_eef.tolist()]}"
            )

        action = np.concatenate([target_pos, target_quat, [gripper_action]]).astype(np.float32)
        return torch.from_numpy(action), target_eef, target_pos, target_quat


    def teleop_step(self, record_data=False):
        if not self.is_connected:
            raise RuntimeError("ScriptRobot is not connected. Run `robot.connect()` first.")

        before_fread_t = time.perf_counter()
        franka_joints = self._get_franka_joints()
        gripper_width = self._get_gripper_width()
        self.logs["read_follower_dt_s"] = time.perf_counter() - before_fread_t

        state = torch.tensor(list(franka_joints) + [gripper_width], dtype=torch.float32)
        action_tensor, _target_eef, target_pos, target_quat = self._scripted_action(state)
        if self._script_step_count % 30 == 0:
            current_eef = self._current_eef_pose(state)
            print(
                "[script] current_joints="
                f"{[round(v, 4) for v in state[:7].tolist()]} "
                f"target_pos={[round(v, 4) for v in action_tensor[:3].tolist()]} "
                f"target_quat={[round(v, 4) for v in action_tensor[3:7].tolist()]} "
                f"current_eef_pos={[round(v, 4) for v in current_eef[6:9].tolist()]} "
                f"current_gripper={state[7].item():.4f} target_gripper={action_tensor[7].item():.4f}"
            )
        self._script_step_count += 1

        pos_diff, angle_diff = self._pose_controller.move_to(
            target_pos=target_pos,
            target_quat=target_quat,
            grasp=None,
            num_steps=50,
            num_additional_steps=20,
            pos_tolerance=self.config.pose_pos_tolerance,
            rot_tolerance=self.config.pose_rot_tolerance,
            max_delta_pos=self.config.pose_max_delta_pos,
            action_smoothing=self.config.pose_action_smoothing,
        )


        before_fwrite_t = time.perf_counter()
        self.logs["write_follower_dt_s"] = time.perf_counter() - before_fwrite_t

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
