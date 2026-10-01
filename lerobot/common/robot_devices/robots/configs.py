# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from dataclasses import dataclass, field
from typing import Sequence

import draccus

from lerobot.common.robot_devices.cameras.configs import (
    AzureKinectCameraConfig,
    CameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.motors.configs import (
    DynamixelMotorsBusConfig,
    FeetechMotorsBusConfig,
    MotorsBusConfig,
)


@dataclass
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


# TODO(rcadene, aliberts): remove ManipulatorRobotConfig abstraction
@dataclass
class ManipulatorRobotConfig(RobotConfig):
    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    # Optionally limit the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length
    # as the number of motors in your follower arms (assumes all follower arms have the same number of
    # motors).
    max_relative_target: list[float] | float | None = None

    # Optionally set the leader arm in torque mode with the gripper motor set to this angle. This makes it
    # possible to squeeze the gripper and have it spring back to an open position on its own. If None, the
    # gripper is not put in torque mode.
    gripper_open_degree: float | None = None

    mock: bool = False

    # save end-effector pose info (enabled for aloha)
    use_eef: bool = False

    def __post_init__(self):
        if self.mock:
            for arm in self.leader_arms.values():
                if not arm.mock:
                    arm.mock = True
            for arm in self.follower_arms.values():
                if not arm.mock:
                    arm.mock = True
            for cam in self.cameras.values():
                if not cam.mock:
                    cam.mock = True

        if self.max_relative_target is not None and isinstance(self.max_relative_target, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(self.max_relative_target):
                    raise ValueError(
                        f"len(max_relative_target)={len(self.max_relative_target)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )


@RobotConfig.register_subclass("aloha")
@dataclass
class AlohaRobotConfig(ManipulatorRobotConfig):
    # Specific to Aloha, LeRobot comes with default calibration files. Assuming the motors have been
    # properly assembled, no manual calibration step is expected. If you need to run manual calibration,
    # simply update this path to ".cache/calibration/aloha"
    calibration_dir: str = ".cache/calibration/aloha_default"

    # /!\ FOR SAFETY, READ THIS /!\
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    # For Aloha, for every goal position request, motor rotations are capped at 5 degrees by default.
    # When you feel more confident with teleoperation or running the policy, you can extend
    # this safety limit and even removing it by setting it to `null`.
    # Also, everything is expected to work safely out-of-the-box, but we highly advise to
    # first try to teleoperate the grippers only (by commenting out the rest of the motors in this yaml),
    # then to gradually add more motors (by uncommenting), until you can teleoperate both arms fully
    max_relative_target: int | None = 5

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                # window_x
                port="/dev/ttyDXL_leader_left",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm430-w350"],
                    "shoulder": [2, "xm430-w350"],
                    "shoulder_shadow": [3, "xm430-w350"],
                    "elbow": [4, "xm430-w350"],
                    "elbow_shadow": [5, "xm430-w350"],
                    "forearm_roll": [6, "xm430-w350"],
                    "wrist_angle": [7, "xm430-w350"],
                    "wrist_rotate": [8, "xl430-w250"],
                    "gripper": [9, "xc430-w150"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                # window_x
                port="/dev/ttyDXL_leader_right",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm430-w350"],
                    "shoulder": [2, "xm430-w350"],
                    "shoulder_shadow": [3, "xm430-w350"],
                    "elbow": [4, "xm430-w350"],
                    "elbow_shadow": [5, "xm430-w350"],
                    "forearm_roll": [6, "xm430-w350"],
                    "wrist_angle": [7, "xm430-w350"],
                    "wrist_rotate": [8, "xl430-w250"],
                    "gripper": [9, "xc430-w150"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_left",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_right",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
        }
    )

    # Troubleshooting: If one of your IntelRealSense cameras freeze during
    # data recording due to bandwidth limit, you might need to plug the camera
    # on another USB hub or PCIe card.
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_high": IntelRealSenseCameraConfig(
                serial_number=128422271347,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_low": IntelRealSenseCameraConfig(
                serial_number=130322270656,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_left_wrist": IntelRealSenseCameraConfig(
                serial_number=218622272670,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_right_wrist": IntelRealSenseCameraConfig(
                serial_number=130322272300,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("koch")
@dataclass
class KochRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0085511",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False


@RobotConfig.register_subclass("koch_bimanual")
@dataclass
class KochBimanualRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch_bimanual"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0085511",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem575E0031751",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem575E0032081",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False


@RobotConfig.register_subclass("moss")
@dataclass
class MossRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/moss"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431091",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("so101")
@dataclass
class So101RobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/so101"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431091",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("so100")
@dataclass
class So100RobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/so100"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431091",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("droid")
@dataclass
class DroidRobotConfig(RobotConfig):
    # GELLO leader arm config
    gello_port: str | None = None  # Auto-detected from /dev/serial/by-id/* if None
    gello_joint_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    # HACK: HARDCODED FOR A SPECIFIC GELLO
    gello_joint_offsets: tuple[float, ...] = (
        3 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
        4 * 3.141592653589793 / 2,
        2 * 3.141592653589793 / 2,
        2 * 3.141592653589793 / 2,
        2 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
    )
    gello_joint_signs: tuple[int, ...] = (1, 1, 1, 1, 1, -1, 1)
    gello_gripper_joint_id: int = 8
    gello_gripper_open_degrees: int = 272
    gello_gripper_close_degrees: int = 234

    # Deoxys / Franka config
    deoxys_general_cfg_file: str = "lerobot/common/robot_devices/robots/franka_configs/charmander_droid.yml"
    deoxys_controller_type: str = "JOINT_IMPEDANCE"
    deoxys_controller_cfg_file: str = "lerobot/common/robot_devices/robots/franka_configs/joint-impedance-controller.yml"

    # Teleop mapping: scale + sign for delta mapping from GELLO to Franka
    mapping_coefficients: tuple[float, ...] = (0.8, -0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
    gripper_threshold: float = 0.5
    gripper_open_action: float = 1.0
    gripper_close_action: float = 0.0

    # Robotiq gripper config
    robotiq_port: str | None = None  # Auto-detected by pyRobotiqGripper if None

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_main": AzureKinectCameraConfig(
                device_id=0,
                fps=30,
                width=1280,
                height=720,
            ),
        }
    )

    # Max joint delta (rad) before smooth interpolation kicks in during teleop.
    # Prevents jerky motion when GELLO drifts between episodes.
    max_safe_joint_delta: float = 0.3

    # Skip interactive GELLO-to-Franka alignment during startup.
    skip_gello_calibration: bool = False

    # save end-effector pose info
    use_eef: bool = True
    mock: bool = False


@RobotConfig.register_subclass("script")
@dataclass
class ScriptRobotConfig(RobotConfig):
    # Direct scripted Franka control via deoxys. This skips GELLO teleoperation.
    deoxys_general_cfg_file: str = "lerobot/common/robot_devices/robots/franka_configs/charmander_droid.yml"
    deoxys_controller_type: str = "JOINT_POSITION"
    deoxys_controller_cfg_file: str = "lerobot/common/robot_devices/robots/franka_configs/joint-position-controller.yml"

    # Cartesian delta applied on every scripted control step, in metres.
    script_delta_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    insertion_pose_path: str = "outputs/scripted_insertion_pose.json"
    z_step: float = 0.0001
    script_mode: str = "pose"
    # Skip policy inference in teleop_step (set via `collect_data.sh --debug`).
    debug: bool = False
    # Task 1
    approach_pos: tuple[float, float, float] = (0.66, -0.012, 0.065)
    target_quat: tuple[float, float, float, float] = (
       9.99670212e-01, 3.90303529e-04, 2.46570828e-02, 7.16535067e-03
    )
    pose_num_steps: int = 200
    pose_num_additional_steps: int = 100
    pose_pos_tolerance: float = 0.004
    pose_rot_tolerance: float = 0.05
    pose_pos_action_gain: float = 20.0
    pose_rot_action_gain: float = 2.0
    pose_max_delta_pos: float = 0.025
    pose_action_smoothing: float = 0.6
    home_joints: tuple[float, ...] = (
        -0.74921682,
        0.13623207,
        0.37435664,
        -2.00871515,
        -0.54053575,
        2.19774203,
        2.34971468,
    )
    max_joint_step: float = 0.002
    script_joint_wait_times: int = 100
    script_joint_convergence_tolerance: float = 1e-3
    script_joint_solution_threshold: float = 0.5
    script_rot_max_angle_step_deg: float = 2.0

    # Initial pose randomization, matching AGOS (third_party/AGOS, AutoMateTaskAGOS):
    # lift along the socket axis, then world-frame xyz noise and RPY + axial-spin noise
    # applied about the fingertip (delta = q_rpy * q_axial, target = delta * aligned).
    # Height (m) of the plug above the aligned pose (plug tip just above the socket hole).
    # AGOS lifts 1.4 * d + 0.05 from the seated pose; for asset 300006 that leaves the plug
    # bottom ~0.083 m above the socket top.
    init_lift_height: float = 0.083
    init_pos_noise: tuple[float, float, float] = (0.01, 0.01, 0.0)
    init_rot_noise_deg: tuple[float, float, float] = (10.0, 10.0, 10.0)
    init_axial_spin_deg: float = 180.0
    # AGOS wrist budget: |total twist about the insertion axis| <= 180 - return margin (30).
    init_max_total_twist_deg: float = 150.0
    init_joint7_limit_margin: float = 0.15

    # Multi-view socket pre-scan, matching AGOS `env.plug_photo.socket_views`: the fingertip is
    # driven to socket_top + offset (world frame, m) keeping the start-pose orientation, one
    # wrist capture per view, and the world-frame clouds are concatenated.
    socket_scan_views: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.16),
        (0.06, 0.0, 0.14),
        (-0.06, 0.0, 0.14),
    )
    # Socket top = aligned_pos - (0, 0, this). aligned_pos is the fingertip with the plug tip just
    # above the hole, so set this to the fingertip-to-plug-tip length to match the sim anchor.
    socket_scan_plug_tip_offset: float = 0.0
    socket_scan_move_steps: int = 10
    # FoundationStereo input scale and unprojection stride for the scan. The default pipeline runs
    # at scale 0.5 (640x360 depth), which leaves the socket with only ~800 points per view.
    socket_scan_stereo_scale: float = 1.0
    socket_scan_stride: int = 1
    # Stand-in for the sim's socket segmentation mask: keep world points in this z band and
    # within this xy radius of the socket (<= 0 disables the radius crop).
    socket_scan_z_range: tuple[float, float] = (0.025, 0.06)
    socket_scan_crop_radius: float = 0.08

    # Plug pre-scan with the upward-looking auxiliary ZED on the table (AGOS capture_plug_bottom_view).
    # The fingertip visits plug_photo_pos + each offset (world, m) keeping the held orientation, so
    # the camera sees the plug tip from several angles (sim: two bottom cameras ~39 deg off vertical).
    plug_photo_pos: tuple[float, float, float] = (0.535, -0.15, 0.22)
    # Centre + a ring of 8 views (sides and diagonals, 6 cm radius) so a tilted plug is seen from
    # every side of the camera.
    plug_photo_views: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.06, 0.0, 0.0),
        (0.0424, 0.0424, 0.0),
        (0.0, 0.06, 0.0),
        (-0.0424, 0.0424, 0.0),
        (-0.06, 0.0, 0.0),
        (-0.0424, -0.0424, 0.0),
        (0.0, -0.06, 0.0),
        (0.0424, -0.0424, 0.0),
    )
    # world_from_aux_cam. The camera optical axes are aligned with world x/y/z (camera on the table
    # looking up, square to the table edges). The plug canonical image is centred on the plug's own
    # mean and its lowest point, so only the rotation matters; the position just places the cloud.
    aux_cam_rot_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    aux_cam_pos_world: tuple[float, float, float] = (0.535, -0.15, 0.03)
    # Estimate the camera rotation from the plug scan itself (the views only translate, so the plug's
    # displacement seen by the camera vs the fingertip displacement gives world_from_aux rotation).
    # aux_cam_rot_euler_deg is then only the fallback.
    aux_cam_estimate_rotation: bool = True
    plug_photo_stereo_scale: float = 1.0
    plug_photo_max_depth_m: float = 0.4
    # Green colour segmentation of the plug (stand-in for the sim plug segmentation id).
    plug_green_hue_range_deg: tuple[float, float] = (70.0, 170.0)
    plug_green_min_saturation: float = 0.25
    plug_green_min_value: float = 0.15
    # Statistical outlier removal on the fused plug cloud (0 neighbours disables).
    plug_outlier_nb_neighbors: int = 20
    plug_outlier_std_ratio: float = 2.0
    # Keep only the largest connected cluster (per view and after fusion): drops detached streaks
    # such as green-tinted finger edges / shadows. Clustering runs on a voxel grid of this size.
    plug_cluster_eps_m: float = 0.004  # <= 0 disables
    plug_cluster_min_points: int = 5
    plug_cluster_voxel_m: float = 0.001

    # Master switch for gravity compensation. False = plain open-loop IK control everywhere
    # (no refinement iterations, no per-step feedforward), for comparison.
    gravity_compensation: bool = True
    # Closed-loop pose refinement: re-command target + measured error so that gravity sag
    # and IK/FK model mismatch do not leave a residual (tilt) error at the EEF.
    pose_refine_max_iters: int = 10
    pose_refine_pos_tol: float = 0.0005
    pose_refine_rot_tol_deg: float = 0.2
    pose_refine_gain: float = 1.0
    pose_refine_settle_steps: int = 20
    pose_refine_max_pos_correction: float = 0.01
    pose_refine_max_rot_correction_deg: float = 5.0
    # teleop_step uses one control call per step with a feedforward correction carried across
    # steps (warm-started by the last refinement); this is its per-step update gain (0 = frozen).
    step_feedforward_gain: float = 0.5

    gripper_threshold: float = 0.5
    gripper_open_action: float = 1.0
    gripper_close_action: float = 0.0
    robotiq_port: str | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    use_eef: bool = True
    mock: bool = False


@RobotConfig.register_subclass("franka_leap")
@dataclass
class FrankaLeapRobotConfig(RobotConfig):
    # GELLO leader arm config
    gello_port: str | None = None  # Auto-detected from /dev/serial/by-id/* if None
    gello_joint_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    # HACK: HARDCODED FOR A SPECIFIC GELLO
    gello_joint_offsets: tuple[float, ...] = (
        4 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
        2 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
        4 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
    )
    gello_joint_signs: tuple[int, ...] = (1, 1, 1, 1, 1, -1, 1)
    gello_gripper_joint_id: int = 8
    gello_gripper_open_degrees: int = 195
    gello_gripper_close_degrees: int = 152

    # Deoxys / Franka config
    deoxys_general_cfg_file: str = "lerobot/common/robot_devices/robots/franka_configs/charmander_leap.yml"
    deoxys_controller_type: str = "JOINT_IMPEDANCE"
    deoxys_controller_cfg_file: str = "lerobot/common/robot_devices/robots/franka_configs/joint-impedance-controller.yml"

    # Teleop mapping
    max_safe_joint_delta: float = 0.3

    # LEAP hand + Manus glove config
    geort_checkpoint_root: str = "/home/leap/Desktop/GeoRT/checkpoint"
    geort_ckpt_tag: str = "sriram_1"

    # save end-effector pose info
    use_eef: bool = True

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_main": AzureKinectCameraConfig(
                device_id=0,
                fps=30,
                width=1280,
                height=720,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("dummy")
@dataclass
class DummyRobotConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})
    use_eef: bool = False
    mock: bool = False


@RobotConfig.register_subclass("stretch")
@dataclass
class StretchRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "navigation": OpenCVCameraConfig(
                camera_index="/dev/hello-nav-head-camera",
                fps=10,
                width=1280,
                height=720,
                rotation=-90,
            ),
            "head": IntelRealSenseCameraConfig(
                name="Intel RealSense D435I",
                fps=30,
                width=640,
                height=480,
                rotation=90,
            ),
            "wrist": IntelRealSenseCameraConfig(
                name="Intel RealSense D405",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # Network Configuration
    ip: str = "192.168.0.193"
    port: int = 5555
    video_port: int = 5556

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": OpenCVCameraConfig(
                camera_index="/dev/video0", fps=30, width=640, height=480, rotation=90
            ),
            "wrist": OpenCVCameraConfig(
                camera_index="/dev/video2", fps=30, width=640, height=480, rotation=180
            ),
        }
    )

    calibration_dir: str = ".cache/calibration/lekiwi"

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0077581",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                    "left_wheel": (7, "sts3215"),
                    "back_wheel": (8, "sts3215"),
                    "right_wheel": (9, "sts3215"),
                },
            ),
        }
    )

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    mock: bool = False
