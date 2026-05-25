"""Shared robot motion and CLI defaults for camera calibration scripts."""

import os
import time

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation, Slerp

from lerobot.common.policies.robot_adapters import AlohaAdapter
from lerobot.common.robot_devices.control_utils import add_eef_pose
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.scripts.yufei_policy_utils import (
    rotation_transfer_6D_to_matrix,
    rotation_transfer_matrix_to_6D,
    transform_to_world_frame,
)

# Per-step rotation for hand-eye calibration (unknown tag-to-gripper needs ~15-25 deg).
CALIB_ROTATION_ANGLE_MIN_RAD = 0.26  # ~15 deg
CALIB_ROTATION_ANGLE_MAX_RAD = 0.44  # ~25 deg
CALIB_MOTION_STEPS = 20
CALIB_SETTLE_TIME_S = 1.0

DEFAULT_ALOHA_CAMERAS_JSON = (
    '{"cam_azure_kinect_back": {"type": "azurekinect", "device_id": 0, "fps": 30, "width": 1280, '
    '"height": 720, "use_transformed_depth": true, "use_transformed_color": true, "use_ir": true, '
    '"wired_sync_mode": "master"}, '
    '"cam_azure_kinect_front": {"type": "azurekinect", "device_id": 1, "fps": 30, "width": 1280, '
    '"height": 720, "use_transformed_depth": true, "use_transformed_color": true, "use_ir": true, '
    '"wired_sync_mode": "subordinate", "subordinate_delay_off_master_usec": 200}, '
    '"cam_wrist": {"type": "intelrealsense", "serial_number": "218622271027", "fps": 30, '
    '"width": 1280, "height": 720, "use_depth": false}}'
)

KINECT_CAMERA_NAMES = ("cam_azure_kinect_front", "cam_azure_kinect_back")
CALIB_RESULTS_DIR = os.path.join("data", "calibration", "calibration_results")
CALIB_NPZ_FILENAMES = {
    "cam_azure_kinect_front": "camcam_azure_kinect_front_calibration.npz",
    "cam_azure_kinect_back": "camcam_azure_kinect_back_calibration.npz",
}

DEFAULT_CLI_ARGS = [
    "--robot.type=aloha",
    "--robot.use_eef=true",
    f"--robot.cameras={DEFAULT_ALOHA_CAMERAS_JSON}",
    "--control.type=record",
    "--control.fps=15",
    "--control.single_task=calibration",
    "--control.repo_id=sriramsk/calibration",
]

INITIAL_JOINT_POSITIONS = {
    "cam_azure_kinect_back": torch.tensor(
        [
            91.2305,
            192.0410,
            192.0410,
            147.6562,
            148.0078,
            4.6582,
            38.9355,
            -6.8555,
            10.4356,
            75.6738,
            109.5996,
            109.9512,
            117.0703,
            117.3340,
            -60.9961,
            69.6973,
            10.8984,
            77.0851,
        ]
    ).float(),
    
    "cam_azure_kinect_front": torch.tensor(
        [
        91.2305, 192.0410, 192.0410, 147.6562, 148.0078,   4.7461,  38.8477,
         -6.8555,  10.4356, 112.6758,  87.1875,  87.5391,  94.5703,  94.8340,
         32.4316,  75.4980, -24.3457,  77.2536
        ]
    ).float(),
}


def load_solved_camera_extrinsics(calib_results_dir=CALIB_RESULTS_DIR):
    """Load T^base_camera from solve_calibration.py outputs."""
    from lerobot.camera_calibration.solve_calibration import load_calibration_results

    extrinsics = {}
    for cam_name, npz_name in CALIB_NPZ_FILENAMES.items():
        calib_path = os.path.join(calib_results_dir, npz_name)
        if os.path.exists(calib_path):
            T_base_from_camera, _ = load_calibration_results(calib_path)
            extrinsics[cam_name] = T_base_from_camera
        else:
            print(f"Warning: calibration file not found for {cam_name}: {calib_path}")
    return extrinsics


def get_control_fps(cfg):
    return getattr(cfg.control, "fps", None) or 15


def make_robot_adapter():
    return AlohaAdapter(action_space="right_eef")


def sample_random_delta_rotation(
    angle_min_rad=CALIB_ROTATION_ANGLE_MIN_RAD,
    angle_max_rad=CALIB_ROTATION_ANGLE_MAX_RAD,
):
    """Sample a rotation about a random axis (magnitude uniform in [min, max])."""
    angle = np.random.uniform(angle_min_rad, angle_max_rad)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * angle)


def _send_action_at_fps(robot, action, fps):
    loop_start = time.perf_counter()
    robot.send_action(action)
    if fps is not None and fps > 0:
        busy_wait(1.0 / fps - (time.perf_counter() - loop_start))


def interpolate_eef_waypoints(current_eef_pose, target_eef_pose, num_steps):
    """Linear position + slerp rotation between two 10-d eef poses."""
    current_eef_pose = np.asarray(current_eef_pose, dtype=np.float64)
    target_eef_pose = np.asarray(target_eef_pose, dtype=np.float64)
    cur_rot = Rotation.from_matrix(rotation_transfer_6D_to_matrix(current_eef_pose[:6]))
    tgt_rot = Rotation.from_matrix(rotation_transfer_6D_to_matrix(target_eef_pose[:6]))
    slerp_interp = Slerp([0.0, 1.0], Rotation.concatenate([cur_rot, tgt_rot]))
    waypoints = []
    for i in range(1, num_steps + 1):
        alpha = i / num_steps
        pos = (1.0 - alpha) * current_eef_pose[6:9] + alpha * target_eef_pose[6:9]
        rot_6d = rotation_transfer_matrix_to_6D(slerp_interp([alpha]).as_matrix())
        gripper = (1.0 - alpha) * current_eef_pose[9] + alpha * target_eef_pose[9]
        waypoints.append(torch.tensor([*rot_6d, *pos, gripper], dtype=torch.float32))
    return waypoints


def move_to_joint_target(robot, target_joints, current_joints, fps, num_steps=15):
    """Smoothly move in joint space."""
    target_joints = target_joints.float()
    current_joints = current_joints.float()
    for i in range(1, num_steps + 1):
        alpha = i / num_steps
        action = (1.0 - alpha) * current_joints + alpha * target_joints
        _send_action_at_fps(robot, action, fps)


def move_eef_with_interpolation(robot, robot_adapter, target_eef_pose, joint_state, fps, num_steps):
    """Move to a target eef pose through interpolated waypoints with chained IK."""
    current_eef_pose = add_eef_pose(joint_state).cpu().numpy()
    waypoints = interpolate_eef_waypoints(current_eef_pose, target_eef_pose, num_steps)
    ik_seed = joint_state.float()
    for waypoint in waypoints:
        joint_action = robot_adapter.transform_action(waypoint, ik_seed)
        _send_action_at_fps(robot, joint_action.squeeze(0), fps)
        ik_seed = joint_action.float()
    time.sleep(CALIB_SETTLE_TIME_S)


def sample_random_target_eef_pose(joint_state):
    """Sample a random eef target relative to the current pose."""
    cur_eef_pose = add_eef_pose(joint_state).cpu().numpy()
    random_delta_pos = np.random.uniform(-0.06, 0.06, size=(3,))
    target_position = cur_eef_pose[6:9] + random_delta_pos
    cur_rotation = Rotation.from_matrix(rotation_transfer_6D_to_matrix(cur_eef_pose[:6]))
    delta_rotation = sample_random_delta_rotation()
    target_rotation = (delta_rotation * cur_rotation).as_matrix()
    target_rotation_6d = rotation_transfer_matrix_to_6D(target_rotation)
    return np.array(
        target_rotation_6d.tolist() + target_position.tolist() + [cur_eef_pose[9]]
    )


def image_to_hwc_rgb(image):
    """Convert observation image to (H, W, 3) RGB in [0, 255]."""
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] < img.shape[1]:
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]
    if img.max() <= 1.0:
        img = (img * 255.0).astype(np.uint8)
    return img.astype(np.uint8)


def get_observation_color_key(observation, cam_name):
    """Prefer depth-aligned transformed_color over raw color."""
    transformed_key = f"observation.images.{cam_name}.transformed_color"
    if transformed_key in observation:
        return transformed_key
    return f"observation.images.{cam_name}.color"


def _get_color_image_for_point_cloud(observation, cam_name, target_hw):
    h_tgt, w_tgt = target_hw
    color_key = get_observation_color_key(observation, cam_name)
    color_hwc = image_to_hwc_rgb(observation[color_key].numpy())
    if color_hwc.shape[0] != h_tgt or color_hwc.shape[1] != w_tgt:
        color_hwc = cv2.resize(color_hwc, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
    return color_hwc


def _point_cloud_to_hwc(pcd_array, observation, cam_name):
    pcd_hwc = np.asarray(pcd_array, dtype=np.float32)
    if pcd_hwc.ndim == 3:
        return pcd_hwc
    if pcd_hwc.ndim == 2 and pcd_hwc.shape[-1] == 3:
        for key_suffix in (".transformed_depth", ".depth"):
            depth_key = f"observation.images.{cam_name}{key_suffix}"
            if depth_key not in observation:
                continue
            depth = np.asarray(observation[depth_key].numpy()).squeeze()
            h, w = depth.shape[:2]
            if h * w == pcd_hwc.shape[0]:
                return pcd_hwc.reshape(h, w, 3)
    raise ValueError(f"Unexpected point cloud shape: {pcd_hwc.shape}")


def get_colored_point_cloud_in_world(
    observation,
    cam_name,
    T_world_from_camera,
    max_depth_m=2.0,
):
    """Build an Open3D point cloud with RGB colors in the robot base frame."""
    pcd_hwc = _point_cloud_to_hwc(
        observation[f"observation.images.{cam_name}.point_cloud"].numpy(),
        observation,
        cam_name,
    )
    color_hwc = _get_color_image_for_point_cloud(
        observation, cam_name, target_hw=pcd_hwc.shape[:2]
    )

    points_cam_m = pcd_hwc / 1000.0
    valid = (
        (points_cam_m[..., 2] > 0.01)
        & (points_cam_m[..., 2] < max_depth_m)
        & np.isfinite(points_cam_m).all(axis=-1)
    )
    points_cam = points_cam_m[valid]
    if points_cam.shape[0] == 0:
        return o3d.geometry.PointCloud()

    colors = color_hwc[valid].astype(np.float64) / 255.0
    points_world = transform_to_world_frame(points_cam, T_world_from_camera)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def crop_point_cloud(pcd, x_range=(-0.2, 0.2), y_range=(-0.2, 0.2), z_min=0.02):
    """Crop an Open3D point cloud to a workspace region in robot base frame."""
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        return pcd
    mask = (
        (points[:, 2] > z_min)
        & (points[:, 0] > x_range[0])
        & (points[:, 0] < x_range[1])
        & (points[:, 1] > y_range[0])
        & (points[:, 1] < y_range[1])
    )
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(points[mask])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        cropped.colors = o3d.utility.Vector3dVector(colors[mask])
    return cropped
