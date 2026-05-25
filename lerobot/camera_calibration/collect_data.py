"""
Collect gripper poses and ArUco tag poses for camera-to-robot calibration.

Example:
```bash
python lerobot/camera_calibration/collect_data.py \
    --robot.type=aloha \
    --robot.use_eef=true \
    --robot.cameras='...' \
    --control.type=record
```

Running without extra CLI args uses defaults from calibration_common.py.
"""
import os
import pickle
import sys
import time

import numpy as np
import torch
from pyk4a.calibration import CalibrationType
from scipy.spatial.transform import Rotation
from termcolor import cprint
from tqdm import tqdm

from lerobot.camera_calibration.calibration_common import (
    CALIB_MOTION_STEPS,
    CALIB_SETTLE_TIME_S,
    DEFAULT_CLI_ARGS,
    INITIAL_JOINT_POSITIONS,
    get_control_fps,
    make_robot_adapter,
    move_eef_with_interpolation,
    move_to_joint_target,
    sample_random_target_eef_pose,
)
from lerobot.camera_calibration.marker_detection import (
    detect_aruco_markers,
    estimate_transformation,
    get_kinect_ir_frame,
)
from lerobot.common.robot_devices.control_configs import ControlPipelineConfig
from lerobot.common.robot_devices.control_utils import add_eef_pose
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.configs import parser
from lerobot.scripts.yufei_policy_utils import rotation_transfer_6D_to_matrix


@parser.wrap()
def move_robot_and_record_data(cfg: ControlPipelineConfig):
    """Move the robot to random poses and record calibration data."""
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    cam_name = "cam_azure_kinect_front"
    # cam_name = "cam_azure_kinect_back"
    camera = robot.cameras[cam_name].camera
    camera_matrix = camera.calibration.get_camera_matrix(CalibrationType.DEPTH)
    dist_coeffs = camera.calibration.get_distortion_coefficients(CalibrationType.DEPTH)

    data = []
    num_movements = 30
    home_joints = INITIAL_JOINT_POSITIONS[cam_name]
    robot_adapter = make_robot_adapter()
    fps = get_control_fps(cfg)

    for _ in tqdm(range(num_movements)):
        observation = robot.capture_observation()
        print("Moving to calibration home pose...")
        move_to_joint_target(
            robot, home_joints, observation["observation.state"].float(), fps, num_steps=50
        )

        observation = robot.capture_observation()
        joint_state = observation["observation.state"].float()
        distance_to_home = np.linalg.norm(joint_state - home_joints)
        cprint(f"Distance to home: {distance_to_home}", "blue")
        if distance_to_home > 6:
            cprint("Distance to home is too large, exiting...", "red")
            return

        time.sleep(CALIB_SETTLE_TIME_S)
        observation = robot.capture_observation()
        joint_state = observation["observation.state"].float()
        target_eef_pose = sample_random_target_eef_pose(joint_state)

        move_eef_with_interpolation(
            robot,
            robot_adapter,
            target_eef_pose,
            joint_state,
            fps,
            num_steps=CALIB_MOTION_STEPS,
        )

        time.sleep(CALIB_SETTLE_TIME_S)

        observation = robot.capture_observation()
        cur_eef_pose = add_eef_pose(observation["observation.state"]).cpu().numpy()
        gripper_pose = np.eye(4)
        gripper_pose[:3, :3] = rotation_transfer_6D_to_matrix(cur_eef_pose[:6])
        gripper_pose[:3, 3] = cur_eef_pose[6:9]
        print(f"Gripper pos: {gripper_pose[:3, 3]}")

        ir_frame = get_kinect_ir_frame(robot.cameras[cam_name], visualize=False)
        if ir_frame is not None:
            corners, ids = detect_aruco_markers(ir_frame, debug=False)
            if ids is not None and len(ids) > 0:
                print("\033[92m" + f"Detected {len(ids)} markers." + "\033[0m")
                transform_matrix = estimate_transformation(
                    corners, ids, camera_matrix, dist_coeffs
                )
                if transform_matrix is not None:
                    data.append((gripper_pose, transform_matrix))
            else:
                print("\033[91m" + "No markers detected." + "\033[0m")
        else:
            print("\033[91m" + "No IR frame captured." + "\033[0m")

    print(f"Recorded {len(data)} data points.")
    os.makedirs("data/calibration", exist_ok=True)
    filepath = f"data/calibration/cam{cam_name}_data.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    return filepath


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(DEFAULT_CLI_ARGS)
    move_robot_and_record_data()
