"""DroidRobot: Franka Panda (via deoxys) + GELLO leader arm integration for LeRobot."""

import glob
import time

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import DroidRobotConfig


class DroidRobot:
    """Robot class for a Franka Panda controlled via deoxys with GELLO teleoperation.

    The follower arm is a Franka Panda controlled through the deoxys C++ controller
    interface over ZMQ. The leader arm is a GELLO (7-DOF Dynamixel) accessed through
    the gello Python package.
    """

    robot_type = "droid"

    def __init__(self, config: DroidRobotConfig | None = None, **kwargs):
        if config is None:
            self.config = DroidRobotConfig(**kwargs)
        else:
            self.config = config

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.robot_interface = None
        self.gello = None
        self.gello_home = None
        self.robot_home = None
        self.is_connected = False
        self.logs = {}
        self.robot_type = self.config.type
        self.use_eef = self.config.use_eef

        self.joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            "gripper",
        ]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            cam_ft.update(cam.config.get_feature_specs(cam_key))
        return cam_ft

    @property
    def motor_features(self) -> dict:
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(self.joint_names),),
                "names": list(self.joint_names),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.joint_names),),
                "names": list(self.joint_names),
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self):
        if self.is_connected:
            raise RuntimeError("DroidRobot is already connected. Do not run `robot.connect()` twice.")

        # Lazy import deoxys
        from deoxys.franka_interface import FrankaInterface
        from deoxys.utils import YamlConfig

        # Initialize Franka interface
        self.robot_interface = FrankaInterface(
            self.config.deoxys_general_cfg_file,
            use_visualizer=False,
        )
        self.controller_cfg = YamlConfig(
            self.config.deoxys_controller_cfg_file
        ).as_easydict()

        # Wait for state buffer to populate
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

        # Initialize GELLO
        self._init_gello()

        # Connect cameras (two-phase init for multi-Azure Kinect setups)
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
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.is_connected = True

        # Run interactive home calibration
        self.run_calibration()

    def _init_gello(self):
        """Initialize the GELLO leader arm."""
        from gello.robots.dynamixel import DynamixelRobot

        port = self.config.gello_port
        if port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            if len(usb_ports) == 0:
                raise ValueError(
                    "No GELLO port found. Please specify gello_port or plug in GELLO."
                )
            port = usb_ports[-1]
            print(f"using port {port}. HACK: One hardcoded port is selected. Needs to be handled properly.")

        self.gello = DynamixelRobot(
            joint_ids=list(self.config.gello_joint_ids),
            joint_offsets=list(self.config.gello_joint_offsets),
            real=True,
            joint_signs=list(self.config.gello_joint_signs),
            port=port,
            gripper_config=(
                self.config.gello_gripper_joint_id,
                self.config.gello_gripper_open_degrees,
                self.config.gello_gripper_close_degrees,
            ),
        )

    def run_calibration(self):
        """Interactive home calibration: user aligns GELLO to Franka's pose."""
        input(
            "\n[DroidRobot] Align GELLO to match the Franka's current pose.\n"
            "Press Enter when ready..."
        )
        self.robot_home = np.array(self.robot_interface._state_buffer[-1].q)
        self.gello_home = np.array(self.gello.get_joint_state())
        print(f"Calibration done. Robot home: {self.robot_home}")

    def _get_franka_joints(self) -> np.ndarray:
        """Read current 7-DOF joint positions from Franka."""
        return np.array(self.robot_interface._state_buffer[-1].q)

    def _get_gripper_width(self) -> float:
        """Read current gripper width from Franka."""
        if len(self.robot_interface._gripper_state_buffer) > 0:
            return float(self.robot_interface._gripper_state_buffer[-1].width)
        return 0.0

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RuntimeError("DroidRobot is not connected. Run `robot.connect()` first.")

        # Read GELLO state (7 arm joints + 1 gripper)
        before_lread_t = time.perf_counter()
        gello_joints = np.array(self.gello.get_joint_state())
        self.logs["read_leader_dt_s"] = time.perf_counter() - before_lread_t

        # Compute Franka target via delta mapping
        coeffs = np.array(self.config.mapping_coefficients)
        robot_target = coeffs * (gello_joints[:7] - self.gello_home[:7]) + self.robot_home

        # Gripper thresholding
        if gello_joints[-1] > self.config.gripper_threshold:
            gripper_action = self.config.gripper_close_action
        else:
            gripper_action = self.config.gripper_open_action

        # Send to Franka — loop until convergence
        # The deoxys controller needs to be fed commands continuously at a high rate;
        # a single control() call followed by a long gap causes oscillation/vibration.
        action = list(robot_target) + [gripper_action]
        before_fwrite_t = time.perf_counter()
        max_iterations = 60
        for _ in range(max_iterations):
            if len(self.robot_interface._state_buffer) > 0:
                joint_error = np.max(np.abs(
                    np.array(self.robot_interface._state_buffer[-1].q) - robot_target
                ))
                if len(self.robot_interface._gripper_state_buffer) > 0:
                    gripper_width = self.robot_interface._gripper_state_buffer[-1].width
                    gripper_state = 0.0 if np.abs(gripper_width) < 0.01 else 1.0
                    gripper_error = np.abs(gripper_state - gripper_action)
                else:
                    gripper_error = 0.0
                if joint_error < 1e-3 and gripper_error < 1e-3:
                    break
            self.robot_interface.control(
                controller_type=self.config.deoxys_controller_type,
                action=action,
                controller_cfg=self.controller_cfg,
            )
        self.logs["write_follower_dt_s"] = time.perf_counter() - before_fwrite_t

        if not record_data:
            return

        # Read Franka state for observation
        before_fread_t = time.perf_counter()
        franka_joints = self._get_franka_joints()
        gripper_width = self._get_gripper_width()
        self.logs["read_follower_dt_s"] = time.perf_counter() - before_fread_t

        state = torch.tensor(
            list(franka_joints) + [gripper_width], dtype=torch.float32
        )
        action_tensor = torch.tensor(action, dtype=torch.float32)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            if type(images[name]) == dict:
                for img_name in images[name].keys():
                    images[name][img_name] = torch.from_numpy(images[name][img_name])
            else:
                images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
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

    def capture_observation(self) -> dict:
        """Read Franka joint state + gripper + camera images. For policy inference."""
        if not self.is_connected:
            raise RuntimeError("DroidRobot is not connected. Run `robot.connect()` first.")

        before_fread_t = time.perf_counter()
        franka_joints = self._get_franka_joints()
        gripper_width = self._get_gripper_width()
        self.logs["read_follower_dt_s"] = time.perf_counter() - before_fread_t

        state = torch.tensor(
            list(franka_joints) + [gripper_width], dtype=torch.float32
        )

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            if type(images[name]) == dict:
                for img_name in images[name].keys():
                    images[name][img_name] = torch.from_numpy(images[name][img_name])
            else:
                images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            if type(images[name]) == dict:
                for img_name in images[name].keys():
                    obs_dict[f"observation.images.{name}.{img_name}"] = images[name][img_name]
            else:
                obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Send an 8-value action tensor (7 joints + gripper) to Franka via deoxys."""
        if not self.is_connected:
            raise RuntimeError("DroidRobot is not connected. Run `robot.connect()` first.")

        action_list = action.tolist()
        joint_target = np.array(action_list[:7])
        gripper_action = action_list[7]
        before_fwrite_t = time.perf_counter()
        max_iterations = 60
        for _ in range(max_iterations):
            if len(self.robot_interface._state_buffer) > 0:
                joint_error = np.max(np.abs(
                    np.array(self.robot_interface._state_buffer[-1].q) - joint_target
                ))
                if len(self.robot_interface._gripper_state_buffer) > 0:
                    gripper_width = self.robot_interface._gripper_state_buffer[-1].width
                    gripper_state = 0.0 if np.abs(gripper_width) < 0.01 else 1.0
                    gripper_error = np.abs(gripper_state - gripper_action)
                else:
                    gripper_error = 0.0
                if joint_error < 1e-3 and gripper_error < 1e-3:
                    break
            self.robot_interface.control(
                controller_type=self.config.deoxys_controller_type,
                action=action_list,
                controller_cfg=self.controller_cfg,
            )
        self.logs["write_follower_dt_s"] = time.perf_counter() - before_fwrite_t

        return action

    def print_logs(self):
        pass

    def disconnect(self):
        if not self.is_connected:
            return

        if self.robot_interface is not None:
            self.robot_interface.close()
            self.robot_interface = None

        self.gello = None
        self.gello_home = None
        self.robot_home = None

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
