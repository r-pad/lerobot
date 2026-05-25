"""
Visualize calibration quality: point cloud, eef frame, and ArUco tag in robot base frame.

Example:
```bash
python lerobot/camera_calibration/evaluate_calibration_error.py
```

Running without extra CLI args uses defaults from calibration_common.py.
"""
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch
from matplotlib import pyplot as plt
from pyk4a.calibration import CalibrationType

from lerobot.camera_calibration.calibration_common import (
    CALIB_MOTION_STEPS,
    CALIB_NPZ_FILENAMES,
    CALIB_RESULTS_DIR,
    DEFAULT_CLI_ARGS,
    INITIAL_JOINT_POSITIONS,
    get_colored_point_cloud_in_world,
    get_control_fps,
    get_observation_color_key,
    image_to_hwc_rgb,
    load_solved_camera_extrinsics,
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
from lerobot.camera_calibration.solve_calibration import estimate_tag_pose, load_calibration_results
from lerobot.common.robot_devices.control_configs import ControlPipelineConfig
from lerobot.common.robot_devices.control_utils import add_eef_pose
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.configs import parser
from lerobot.scripts.yufei_policy_utils import rotation_transfer_6D_to_matrix

default_extrinsics = load_solved_camera_extrinsics(CALIB_RESULTS_DIR)
tag_to_gripper_by_cam = {}
for cam_name, npz_name in CALIB_NPZ_FILENAMES.items():
    calib_path = os.path.join(CALIB_RESULTS_DIR, npz_name)
    if os.path.exists(calib_path):
        _, T_gripper_from_tag = load_calibration_results(calib_path)
        if T_gripper_from_tag is not None:
            tag_to_gripper_by_cam[cam_name] = T_gripper_from_tag


def get_aruco_coord(camera_device, cam_name):
    """Detect ArUco in IR and return a coordinate frame mesh in robot base frame."""
    ir_frame = get_kinect_ir_frame(camera_device)
    if ir_frame is None:
        print("No IR frame captured.")
        return None

    corners, ids = detect_aruco_markers(ir_frame, debug=False)
    if ids is None or len(ids) == 0:
        print("No ArUco markers detected.")
        return None

    k4a = camera_device.camera
    camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.DEPTH)
    dist_coeffs = k4a.calibration.get_distortion_coefficients(CalibrationType.DEPTH)
    pose_in_cam = estimate_transformation(corners, ids, camera_matrix, dist_coeffs)
    if pose_in_cam is None:
        return None

    aruco_coord_in_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    aruco_coord_in_cam.transform(pose_in_cam)
    return aruco_coord_in_cam.transform(default_extrinsics[cam_name])


@parser.wrap()
def visualize(cfg: ControlPipelineConfig):
    cam_name = "cam_azure_kinect_front"
    # cam_name = "cam_azure_kinect_back"

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot_adapter = make_robot_adapter()
    fps = get_control_fps(cfg)
    home_joints = INITIAL_JOINT_POSITIONS[cam_name]

    observation = robot.capture_observation()
    print("Initial state:", observation["observation.state"])
    rgb = image_to_hwc_rgb(observation[get_observation_color_key(observation, cam_name)].numpy())
    plt.imshow(rgb)
    plt.show()

    # print("Moving to calibration home pose...")
    # move_to_joint_target(
    #     robot, home_joints, observation["observation.state"].float(), fps, num_steps=20
    # )
    # time.sleep(1.0)

    for _ in range(10):
        move_to_joint_target(
            robot, home_joints, observation["observation.state"].float(), fps, num_steps=50
        )
        time.sleep(0.5)

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

        observation = robot.capture_observation()
        colored_pcd = get_colored_point_cloud_in_world(
            observation, cam_name, default_extrinsics[cam_name]
        )

        eef_pose = add_eef_pose(observation["observation.state"]).cpu().numpy()
        eef_pose_matrix = np.eye(4)
        eef_pose_matrix[:3, :3] = rotation_transfer_6D_to_matrix(eef_pose[:6])
        eef_pose_matrix[:3, 3] = eef_pose[6:9]
        print("eef_pose", np.round(eef_pose, 2))

        eef_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        eef_coord.transform(eef_pose_matrix)

        tag_to_gripper = tag_to_gripper_by_cam.get(cam_name)
        _, tag_pose = estimate_tag_pose(eef_pose_matrix, tag_to_gripper=tag_to_gripper)
        aruco_coord_robot = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        aruco_coord_robot.transform(tag_pose) ### this is from the robot fk and the estimated tag to gripper transform

        aruco_coord_cam = get_aruco_coord(robot.cameras[cam_name], cam_name) 
        ### this is using the detection in cam frame and then transformed to the robot base frame using the estimated camera to base transform

        geometries = [colored_pcd, eef_coord, aruco_coord_robot]
        if aruco_coord_cam is not None:
            geometries.append(aruco_coord_cam) 

        o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(DEFAULT_CLI_ARGS)
    visualize()
