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
        self.robotiq_gripper = None
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
        motor_features = {
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

        if self.use_eef:
            eef_names = ["rot_6d_0", "rot_6d_1", "rot_6d_2", "rot_6d_3", "rot_6d_4", "rot_6d_5", "trans_0", "trans_1", "trans_2", "gripper_articulation"]
            motor_features["observation.right_eef_pose"] = {
                "dtype": "float32",
                "shape": (len(eef_names),),
                "names": eef_names,
            }
            motor_features["action.right_eef_pose"] = {
                "dtype": "float32",
                "shape": (len(eef_names),),
                "names": eef_names,
            }

        return motor_features

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

        # Initialize Robotiq gripper (must happen after GELLO so we can exclude its port)
        self._init_robotiq_gripper()

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
            port = usb_ports[0]
            print(f"using port {port}. HACK: One hardcoded port is selected. Needs to be handled properly.")

        self._gello_port = port

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

    def _init_robotiq_gripper(self):
        """Initialize and activate the Robotiq gripper via pyRobotiqGripper.

        When robotiq_port is not set, auto-detects the Robotiq port by scanning
        serial ports while explicitly skipping the GELLO port. This avoids
        pyRobotiqGripper's default auto-detection which probes all ports with
        Modbus RTU packets and corrupts the GELLO Dynamixel communication.
        """
        from pyrobotiqgripper import RobotiqGripper

        port = self.config.robotiq_port
        if port is None:
            port = self._find_robotiq_port()
        print(f"Connecting to Robotiq gripper on {port}")
        self.robotiq_gripper = RobotiqGripper(portname=port)
        print("Activating Robotiq gripper (will fully open/close during activation)...")
        self.robotiq_gripper.activate()
        print("Robotiq gripper activated.")

    def _find_robotiq_port(self) -> str:
        """Find the Robotiq gripper serial port, excluding the GELLO port.

        Scans all serial ports and resolves symlinks so that /dev/serial/by-id/*
        paths (used by GELLO) are correctly matched against /dev/ttyUSB* paths
        (returned by pyserial).
        """
        import os
        import serial.tools.list_ports

        # Resolve the GELLO port symlink to its real device path (e.g. /dev/ttyUSB0)
        gello_real = os.path.realpath(self._gello_port)

        for port_info in serial.tools.list_ports.comports():
            port_real = os.path.realpath(port_info.device)
            if port_real == gello_real:
                continue
            # Probe this port for a Robotiq gripper
            try:
                import minimalmodbus as mm
                import serial

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
            f"or check that the gripper is connected. (GELLO is on {self._gello_port})"
        )

    def run_calibration(self):
        """Move Franka to match the GELLO's current pose for absolute control."""
        input(
            "\n[DroidRobot] Position the GELLO at your desired starting pose.\n"
            "Press Enter when ready — the Franka will move to match..."
        )

        gello_joints = np.array(self.gello.get_joint_state())
        gello_target = gello_joints[:7]
        franka_current = np.array(self.robot_interface._state_buffer[-1].q)

        print(f"  Franka current: {np.round(franka_current, 4)}")
        print(f"  GELLO target:   {np.round(gello_target, 4)}")
        print(f"  Moving Franka to match GELLO...")

        # Interpolate smoothly from current Franka pose to GELLO pose
        max_delta = np.max(np.abs(gello_target - franka_current))
        num_steps = max(int(max_delta / 0.01), 1)  # ~0.01 rad per step
        for i in range(num_steps):
            alpha = (i + 1) / num_steps
            waypoint = franka_current + alpha * (gello_target - franka_current)
            deoxys_action = list(waypoint) + [self.config.gripper_open_action]
            self.robot_interface.control(
                controller_type=self.config.deoxys_controller_type,
                action=deoxys_action,
                controller_cfg=self.controller_cfg,
            )
        print(f"[DroidRobot] Franka aligned to GELLO. Ready for teleop.")

    def _get_franka_joints(self) -> np.ndarray:
        """Read current 7-DOF joint positions from Franka."""
        return np.array(self.robot_interface._state_buffer[-1].q)

    def _get_gripper_width(self) -> float:
        """Read current gripper position from Robotiq, normalized to [0, 1].

        Returns 1.0 for fully open, 0.0 for fully closed.
        Robotiq convention: position 0 = open, 255 = closed.
        """
        pos = self.robotiq_gripper.getPosition()  # 0 (open) to 255 (closed)
        return 1.0 - (pos / 255.0)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RuntimeError("DroidRobot is not connected. Run `robot.connect()` first.")

        # Read GELLO state (7 arm joints + 1 gripper)
        before_lread_t = time.perf_counter()
        gello_joints = np.array(self.gello.get_joint_state())
        self.logs["read_leader_dt_s"] = time.perf_counter() - before_lread_t

        # Absolute control: GELLO is a kinematic replica of Franka, so joint
        # angles map 1:1 (offsets/signs already applied by DynamixelRobot)
        robot_target = gello_joints[:7]

        # Gripper thresholding
        if gello_joints[-1] > self.config.gripper_threshold:
            gripper_action = self.config.gripper_close_action
        else:
            gripper_action = self.config.gripper_open_action

        # Send gripper command to Robotiq only when state changes (avoid blocking
        # Modbus RTU round-trips on every tick — open()/close() take ~30ms each)
        if not hasattr(self, '_last_gripper_action') or gripper_action != self._last_gripper_action:
            if gripper_action == self.config.gripper_close_action:
                self.robotiq_gripper.close()
            else:
                self.robotiq_gripper.open()
            self._last_gripper_action = gripper_action

        # Send joint target to Franka via deoxys
        deoxys_action = list(robot_target) + [gripper_action]
        action = list(robot_target) + [gripper_action]
        before_fwrite_t = time.perf_counter()
        self.robot_interface.control(
            controller_type=self.config.deoxys_controller_type,
            action=deoxys_action,
            controller_cfg=self.controller_cfg,
        )
        deoxys_dt = time.perf_counter() - before_fwrite_t
        self.logs["write_follower_dt_s"] = deoxys_dt

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

        # Threshold continuous gripper value into open/close
        if action_list[7] < self.config.gripper_threshold:
            gripper_action = self.config.gripper_close_action
        else:
            gripper_action = self.config.gripper_open_action

        # Send gripper command to Robotiq only on change
        if not hasattr(self, '_last_gripper_action') or gripper_action != self._last_gripper_action:
            if gripper_action == self.config.gripper_close_action:
                self.robotiq_gripper.close()
            else:
                self.robotiq_gripper.open()
            self._last_gripper_action = gripper_action

        # Send joint target to Franka via deoxys
        deoxys_action = list(joint_target) + [gripper_action]
        self.robot_interface.control(
            controller_type=self.config.deoxys_controller_type,
            action=deoxys_action,
            controller_cfg=self.controller_cfg,
        )
        self.logs["write_follower_dt_s"] = 0.0

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
        self.robotiq_gripper = None

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
