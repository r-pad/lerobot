"""
LeRobotDataset doesn't provide any easy way to modify the dataset post-hoc with new keys.
This script loads and iters through a dataset, does any extra processing as necessary,
and creates a new dataset.

Slow, but flexible.
"""
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
from tqdm import tqdm
import numpy as np
from lerobot.common.utils.aloha_utils import render_aloha_gripper_pcd, retarget_aloha_gripper_pcd
from lerobot.common.policies.high_level.classify_utils import TASK_SPEC
from PIL import Image
from typing import List, Dict
import argparse
import os
import imageio.v3 as iio
import pytorch3d.transforms as transforms
import json
from typing import Optional
import av
from pathlib import Path
from torchvision import transforms
from pytorch3d.ops import sample_farthest_points
import PIL


TARGET_SHAPE = 224
depth_preprocess = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SHAPE,
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.CenterCrop(TARGET_SHAPE),
    ]
)


def depth_to_pointcloud(depth, K, cam_to_world):
    """
    depth: [H, W] depth map (meters)
    K: [3, 3] intrinsics
    cam_to_world: [4, 4] camera pose matrix
    """
    device = depth.device
    H, W = depth.shape

    # Make meshgrid
    u, v = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy"
    )

    # Unproject to camera coordinates
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack and homogenize
    pc_cam = torch.stack((x, y, z, torch.ones_like(z)), dim=-1).float()  # [H, W, 4]
    pc_cam = pc_cam.reshape(-1, 4)  # [HW, 4]

    # Transform to world
    pc_world = (torch.tensor(cam_to_world).float() @ pc_cam.T).T[:, :3]  # [HW, 3]

    return pc_world


def get_scaled_intrinsics(K, orig_shape, target_shape):
    """
    Scale camera intrinsics based on image resizing and cropping.

    Args:
        K (np.ndarray): Original 3x3 camera intrinsic matrix.
        orig_shape (tuple): Original image shape (height, width).
        target_shape (int): Target size for resize and crop.

    Returns:
        np.ndarray: Scaled 3x3 intrinsic matrix.
    """
    # Getting scale factor from torchvision.transforms.Resize behaviour
    K_ = K.copy()

    scale_factor = target_shape / min(orig_shape)

    # Apply the scale factor to the intrinsics
    K_[0, 0] *= scale_factor  # fx
    K_[1, 1] *= scale_factor  # fy
    K_[0, 2] *= scale_factor  # cx
    K_[1, 2] *= scale_factor  # cy

    # Adjust the principal point (cx, cy) for the center crop
    crop_offset_x = (orig_shape[1] * scale_factor - target_shape) / 2
    crop_offset_y = (orig_shape[0] * scale_factor - target_shape) / 2

    # Adjust the principal point (cx, cy) for the center crop
    K_[0, 2] -= crop_offset_x  # Adjust cx for crop
    K_[1, 2] -= crop_offset_y  # Adjust cy for crop
    return K_


def compute_pcd(depth, K, num_points, max_depth, cam_to_world):
    """
    Compute a downsampled point cloud from RGB and depth images.

    Args:
        rgb (np.ndarray): RGB image array (H, W, 3). np.uint8
        depth (np.ndarray): Depth image array (H, W). np.uint16
        K (np.ndarray): 3x3 camera intrinsic matrix.
        rgb_preprocess (transforms.Compose): Preprocessing for RGB.
        depth_preprocess (transforms.Compose): Preprocessing for depth.
        device (torch.device): Device for computations.
        rng (np.random.Generator): Random number generator.
        num_points (int): Number of points to sample.
        max_depth (float): Maximum depth threshold.

    Returns:
        np.ndarray: Downsampled point cloud (N, 6) with XYZ and RGB.
    """
    depth_ = (depth.numpy() / 1000.0).squeeze().astype(np.float32)
    depth_ = PIL.Image.fromarray(depth_)
    depth_ = np.asarray(depth_preprocess(depth_))

    height, width = depth_.shape
    # Create pixel coordinate grid
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Flatten grid coordinates and depth
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = depth_.flatten()

    # Remove points with invalid depth
    valid_depth = np.logical_and(z_flat > 0, z_flat < max_depth)
    x_flat = x_flat[valid_depth]
    y_flat = y_flat[valid_depth]
    z_flat = z_flat[valid_depth]

    # Create homogeneous pixel coordinates
    pixels = np.stack([x_flat, y_flat, np.ones_like(x_flat)], axis=0)

    # Unproject points using K inverse
    K_inv = np.linalg.inv(K)
    points = K_inv @ pixels
    points = points * z_flat
    points = points.T  # Shape: (N, 3)

    scene_pcd_pt3d = torch.from_numpy(points)
    scene_pcd_downsample, scene_points_idx = sample_farthest_points(
        scene_pcd_pt3d[None], K=num_points, random_start_point=False
    )
    scene_pcd = scene_pcd_downsample.squeeze().numpy()

    pcd_xyz_hom = np.concatenate([scene_pcd, np.ones((scene_pcd.shape[0], 1))], axis=1)  # (N, 4)
    pcd_xyz_world = (cam_to_world @ pcd_xyz_hom.T).T[:, :3]  # (N, 3)

    # Get corresponding colors at the indices
    return pcd_xyz_world


def load_calibrations(calibration_config_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load camera calibrations from JSON config.

    JSON format:
    {
        "cam_azure_kinect": {
            "intrinsics": "path/to/intrinsics.txt",
            "extrinsics": "path/to/extrinsics.txt"
        }
    }

    Returns: Dict mapping camera names to {"K": intrinsics, "T_world_cam": extrinsics}
    """
    with open(calibration_config_path, 'r') as f:
        config = json.load(f)

    calibrations = {}
    for cam_name, cam_config in config.items():
        calibrations[cam_name] = {
            "K": np.loadtxt(cam_config["intrinsics"]),
            "T_world_cam": np.loadtxt(cam_config["extrinsics"])
        }

    print(f"Loaded calibrations for {len(calibrations)} camera(s): {list(calibrations.keys())}")
    return calibrations


def read_depth_video(video_path):
    # Open video with PyAV directly
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    loaded_frames = []
    # Decode frames
    for frame in container.decode(video_stream):
        # Convert frame to numpy array preserving bit depth
        if frame.format.name in ["gray16le", "gray16be"]:
            # 16-bit grayscale
            frame_array = frame.to_ndarray(format="gray16le") / 1000.0
        else:
            raise NotImplementedError("Not supporting other formats right now.")
        loaded_frames.append(frame_array)
    container.close()

    frames = np.stack([frame for frame in loaded_frames])
    return frames

def extract_events_with_gripper_pos(
    joint_states, close_thresh=15, open_thresh=25
):
    """
    Extract all gripper open/close events dynamically.
    Each time the gripper closes and then opens constitutes one goal.
    The last frame is also considered a goal.
    Returns list of goal frame indices.
    """
    gripper_pos = joint_states[:, 17]
    goal_indices = []

    # Track gripper state changes
    is_closed = False

    for i in range(len(gripper_pos)):
        # Gripper closes (transition from open to closed)
        if not is_closed and gripper_pos[i] < close_thresh:
            is_closed = True
            goal_indices.append(i)
        # Gripper opens (transition from closed to open)
        elif is_closed and gripper_pos[i] > open_thresh:
            is_closed = False
            goal_indices.append(i)

    # Always add the last frame as a goal
    if len(goal_indices) == 0 or goal_indices[-1] != len(gripper_pos) - 1:
        goal_indices.append(len(gripper_pos) - 1)

    return goal_indices

def prep_eef_pose(eef_pos, eef_rot, eef_artic):
    REAL_GRIPPER_MIN, REAL_GRIPPER_MAX = 0., 99.
    SIM_GRIPPER_MIN, SIM_GRIPPER_MAX = 0., 0.041
    eef_artic = (eef_artic - SIM_GRIPPER_MIN)*(
        (REAL_GRIPPER_MAX-REAL_GRIPPER_MIN)/(SIM_GRIPPER_MAX-SIM_GRIPPER_MIN)
    ) + REAL_GRIPPER_MIN
    rot_6d = transforms.matrix_to_rotation_6d(torch.from_numpy(eef_rot)).numpy()
    eef_pose = np.concatenate([rot_6d, eef_pos, eef_artic[:, None]], axis=1).astype(np.float32)
    return eef_pose

def get_goal_image(K, width, height, four_points=True, goal_repr="heatmap", humanize=False, gripper_pcd=None, cam_to_world=None, joint_state=None):
    """
    Generate goal image from robot/human hand point cloud data

    Robot: Render gripper pcd in camera frame, project the full point cloud or 4 handpicked points to a mask
    Hand: Extracted with WiLoR and postprocessed to be in world frame
    """
    if not humanize:
        mesh = render_aloha_gripper_pcd(cam_to_world=cam_to_world, joint_state=joint_state)
        gripper_idx = np.array([6, 197, 174]) # Handpicked idxs
    else:
        mesh = gripper_pcd
        gripper_idx = np.array([343, 763, 60]) # Handpicked idxs

        mesh_hom = np.concatenate(
            [mesh, np.ones((mesh.shape[0], 1))], axis=-1
        )[:, :, None]
        world_to_cam = np.linalg.inv(cam_to_world)
        mesh = (world_to_cam @ mesh_hom)[:, :3].squeeze(2)

    assert goal_repr in ["mask", "heatmap"]

    if four_points:
        mesh_primary_points = mesh[gripper_idx]
        # Assumes 0/1 are tips to be averaged
        mesh_extra_point = (mesh_primary_points[0] + mesh_primary_points[1]) / 2
        mesh = np.concatenate([mesh_primary_points, mesh_extra_point[None]])
    else:
        pass

    urdf_proj_hom = (K @ mesh.T).T
    urdf_proj = (urdf_proj_hom / urdf_proj_hom[:, 2:])[:, :2]
    urdf_proj = np.clip(urdf_proj, [0, 0], [width - 1, height - 1]).astype(int)

    goal_image = np.zeros((height, width, 3))
    if goal_repr == "mask":
        goal_image[urdf_proj[:, 1], urdf_proj[:, 0]] = 255
    elif goal_repr == "heatmap":
        max_distance = np.sqrt(width**2 + height**2)
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        pixel_coords = np.stack([x_coords, y_coords], axis=-1)
        for i in range(3):
            target_point = urdf_proj[i]  # (2,)
            distances = np.linalg.norm(pixel_coords - target_point, axis=-1)  # (height, width)
            goal_image[:, :, i] = distances

        # Apply square root transformation for steeper near-target gradients
        goal_image = (np.sqrt(goal_image / max_distance) * 255)
        goal_image = np.clip(goal_image, 0, 255).astype(np.uint8)
    return goal_image

def _load_episode_extras(episode_idx, phantomize, humanize, path_to_extradata, camera_names):
    """Load phantom retargeted data or human demo extras for an episode."""
    episode_extras = {}

    if phantomize:
        PHANTOM_VID_DIR = f"{path_to_extradata}/phantom_retarget"
        EVENTS_DIR = f"{path_to_extradata}/events"

        vid_dirs = [
            f"{PHANTOM_VID_DIR}/episode_{episode_idx:06d}_{cam_name}.mp4/"
            for cam_name in camera_names
        ]
        assert all([os.path.exists(i) for i in vid_dirs])

        events_file = f"{EVENTS_DIR}/episode_{episode_idx:06d}.mp4.json"
        assert os.path.exists(events_file)
        with open(events_file, 'r') as f:
            episode_events = json.load(f)
        event_idxs = episode_events
        phantom_vid = {
            cam_name: torch.from_numpy(iio.imread(f"{vid_dir}/episode_{episode_idx:06d}_{cam_name}.mp4"))
            for vid_dir, cam_name in zip(vid_dirs, camera_names)
        }
        phantom_depth_vid = {
            cam_name: torch.from_numpy(read_depth_video(f"{vid_dir}/depth_episode_{episode_idx:06d}_{cam_name}.mkv"))
            for vid_dir, cam_name in zip(vid_dirs, camera_names)
        }

        # Use vid_dirs[0] as it's the same for all cameras
        phantom_proprio = np.load(f"{vid_dirs[0]}/episode_{episode_idx:06d}_{camera_names[0]}_eef.npz")
        phantom_eef_pose = torch.from_numpy(prep_eef_pose(phantom_proprio["eef_pos"],
                                         phantom_proprio["eef_rot"],
                                         phantom_proprio["eef_artic"]))
        phantom_joint_state = torch.from_numpy(phantom_proprio["joint_state"]).float()

        episode_extras.update({
            'phantom_vid': phantom_vid,
            'phantom_depth_vid': phantom_depth_vid,
            'phantom_eef_pose': phantom_eef_pose,
            'phantom_joint_state': phantom_joint_state,
            'episode_events': event_idxs,
        })

    elif humanize:
        EVENTS_DIR = f"{path_to_extradata}/events"
        GRIPPER_PCDS_DIR = f"{path_to_extradata}/wilor_hand_pose"

        events_file = f"{EVENTS_DIR}/episode_{episode_idx:06d}.mp4.json"
        with open(events_file, 'r') as f:
            episode_events = json.load(f)

        gripper_pcds_file = f"{GRIPPER_PCDS_DIR}/episode_{episode_idx:06d}.mp4.npy"
        episode_gripper_pcds = np.load(gripper_pcds_file).astype(np.float32)

        episode_extras.update({
            'episode_events': episode_events,
            'episode_gripper_pcds': episode_gripper_pcds
        })

    else:
        EVENTS_DIR = f"{path_to_extradata}/events"
        events_file = f"{EVENTS_DIR}/episode_{episode_idx:06d}.mp4.json"
        if os.path.exists(events_file):
            with open(events_file, 'r') as f:
                episode_extras['episode_events'] = json.load(f)
    return episode_extras


def _process_frame_data(original_frame, source_dataset, expanded_features, source_meta,
                       phantomize, humanize, episode_extras, frame_idx, episode_length,
                       new_features, calibrations):
    """Process a single frame's data with additional features."""
    frame_data = {}

    # Define fields that LeRobot manages automatically
    AUTO_FIELDS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}

    # Copy existing data
    for key in source_dataset.features.keys():
        if key not in AUTO_FIELDS and key in expanded_features:
            frame_data[key] = original_frame[key]

    # Add embodiment name
    if humanize: frame_data["embodiment"] = "human"
    else: frame_data["embodiment"] = "aloha"

    frame_data["task"] = source_meta.tasks[original_frame['task_index'].item()]
    camera_names = list(calibrations.keys())
    rgb_data, depth_data = {}, {}

    if phantomize:
        for cam_name in camera_names:
            rgb_data[cam_name] = episode_extras[f'phantom_vid'][cam_name][frame_idx]
            depth_data[cam_name] = (episode_extras[f'phantom_depth_vid'][cam_name][frame_idx][:, :, None] * 1000).to(torch.uint16)
        eef_data = episode_extras['phantom_eef_pose'][frame_idx]
        joint_state = episode_extras['phantom_joint_state'][frame_idx]
        next_idx = (frame_idx + 1) if (frame_idx + 1) < episode_length else frame_idx
        action_eef = episode_extras['phantom_eef_pose'][next_idx]
        action = episode_extras['phantom_joint_state'][next_idx]
        goal_indices = episode_extras['episode_events']['event_idxs']
    else:
        # If robot data, we copy over the original robot states/actions
        # If human data without retargeting, we don't have any robot states/actions
        # and so we just copy over the original states/actions as a placeholder.
        for cam_name in camera_names:
            rgb_data[cam_name] = (frame_data[f"observation.images.{cam_name}.color"].permute(1,2,0) * 255).to(torch.uint8)
            depth_data[cam_name]= (frame_data[f"observation.images.{cam_name}.transformed_depth"].permute(1,2,0) * 1000).to(torch.uint16)
        eef_data = frame_data["observation.right_eef_pose"]
        joint_state = frame_data["observation.state"]
        action_eef = frame_data["action.right_eef_pose"]
        action = frame_data["action"]

    for cam_name in camera_names:
        frame_data[f"observation.images.{cam_name}.color"] = rgb_data[cam_name]
        frame_data[f"observation.images.{cam_name}.transformed_depth"] = depth_data[cam_name]

    if 'observation.images.cam_wrist' in frame_data:
        wrist_rgb_data = (frame_data["observation.images.cam_wrist"].permute(1,2,0) * 255).to(torch.uint8)
        frame_data["observation.images.cam_wrist"] = wrist_rgb_data

    frame_data["observation.right_eef_pose"] = eef_data
    frame_data["observation.state"] = joint_state
    frame_data["action.right_eef_pose"] = action_eef
    frame_data["action"] = action

    if "observation.points.gripper_pcds" in new_features:
        if humanize:
            # We've made the switch to keeping gripper_pcds in robot frame, so the detected hand pcd also needs to be transformed to the robot frame.
            frame_data["observation.points.gripper_pcds"] = episode_extras['episode_gripper_pcds'][frame_idx]
        else:
            # Keep in world frame
            frame_data["observation.points.gripper_pcds"] = render_aloha_gripper_pcd(cam_to_world=np.eye(4), joint_state=joint_state).astype(np.float32)

    # Add calibration data if feature is enabled
    for cam_name in camera_names:
        intrinsics_key = f"observation.{cam_name}.intrinsics"
        extrinsics_key = f"observation.{cam_name}.extrinsics"

        if intrinsics_key in new_features:
            frame_data[intrinsics_key] = torch.from_numpy(calibrations[cam_name]["K"]).float()
        if extrinsics_key in new_features:
            frame_data[extrinsics_key] = torch.from_numpy(calibrations[cam_name]["T_world_cam"]).float()

    if "observation.points.point_cloud" in new_features:
        all_pcd = []
        for cam_name in camera_names:
            scaled_K = calibrations[cam_name]["scaled_K"]
            cam_to_world = calibrations[cam_name]["T_world_cam"]
            depth = frame_data["observation.images.{}.transformed_depth".format(cam_name)].squeeze()
            pcd = compute_pcd(depth, scaled_K, 4500, 1.5, cam_to_world)
            all_pcd.append(pcd)

        all_pcd = np.concatenate(all_pcd, axis=0)
        scene_pcd_pt3d = torch.from_numpy(all_pcd)
        # print(all_pcd.shape)
        scene_pcd_downsample, scene_points_idx = sample_farthest_points(
            scene_pcd_pt3d[None], K=4500, random_start_point=False
        )
        scene_pcd = scene_pcd_downsample.squeeze()
        frame_data["observation.points.point_cloud"] = scene_pcd

    # Dummy values, replaced at the end of the episode
    goal_key = f"observation.points.goal_gripper_pcds"
    if goal_key in new_features:
        frame_data[goal_key] = torch.zeros_like(frame_data[f"observation.points.gripper_pcds"])

    if "next_event_idx" in new_features:
        frame_data["next_event_idx"] = np.array([0], dtype=np.int32)
    else:
        frame_data["next_event_idx"] = frame_data["next_event_idx"].numpy().astype(np.int32).reshape(-1,)
    if "subgoal" in new_features:
        frame_data["subgoal"] = ""
    
    for cam_name in camera_names:
        goal_key = f"observation.images.{cam_name}.goal_gripper_proj"
        if goal_key in new_features:
            frame_data[goal_key] = frame_data[goal_key] = torch.zeros_like(frame_data[f"observation.images.{cam_name}.color"])

    return frame_data


def _process_episode_goals(target_dataset, episode_length, new_features, humanize,
                          episode_extras, phantomize, calibrations, width, height):
    """Process goal projections, event indices, and subgoals for an episode."""
    joint_states = np.concatenate([target_dataset.episode_buffer['observation.state']])

    if humanize:
        episode_gripper_pcds = episode_extras['episode_gripper_pcds']

    # Determine goal indices
    if 'episode_events' in episode_extras:
        # Use events from external file (human data, phantom, or robot with manual annotation)
        goal_indices = episode_extras['episode_events']['event_idxs']
        # Ensure last frame is included as a goal
        if goal_indices[-1] != episode_length - 1:
            goal_indices = goal_indices + [episode_length - 1]
    else:
        # Fall back to gripper-based detection for robot data
        close_thresh, open_thresh = 25, 30
        goal_indices = extract_events_with_gripper_pos(
            joint_states, close_thresh=close_thresh, open_thresh=open_thresh)

    camera_names = list(calibrations.keys())

    # Check if any camera has goal_gripper_proj feature
    has_goal_proj = any(f"observation.images.{cam_name}.goal_gripper_proj" in new_features
                        for cam_name in camera_names)

    if has_goal_proj:
        # Generate goal images for each camera at each goal index
        for cam_name in camera_names:
            goal_key = f"observation.images.{cam_name}.goal_gripper_proj"
            K = calibrations[cam_name]["K"]
            cam_to_world = calibrations[cam_name]["T_world_cam"]

            goal_images = []
            for goal_idx in goal_indices:
                if humanize:
                    goal_img = get_goal_image(K, width, height, humanize=True, gripper_pcd=episode_gripper_pcds[goal_idx], cam_to_world=cam_to_world)
                else:
                    goal_img = get_goal_image(K, width, height, cam_to_world=cam_to_world, joint_state=joint_states[goal_idx])
                goal_images.append(Image.fromarray(goal_img).convert("RGB"))

            # Assign goal images to frames based on segments
            current_goal_idx = 0
            for i in range(episode_length):
                # Move to next goal if we've passed the current goal index
                if current_goal_idx < len(goal_indices) - 1 and i >= goal_indices[current_goal_idx]:
                    current_goal_idx += 1

                goal_images[current_goal_idx].save(target_dataset.episode_buffer[goal_key][i])

    if "next_event_idx" in new_features:
        current_goal_idx = 0
        for i in range(episode_length):
            # Find the next goal index for this frame
            while current_goal_idx < len(goal_indices) - 1 and i >= goal_indices[current_goal_idx]:
                current_goal_idx += 1

            target_dataset.episode_buffer["next_event_idx"][i] = goal_indices[current_goal_idx]

    if "subgoal" in new_features:
        task = target_dataset.episode_buffer['task'][0]
        # Assert that the number of tasks in spec matches the number of detected events
        assert task in TASK_SPEC, f"Task '{task}' not found in TASK_SPEC"
        assert len(TASK_SPEC[task]) == len(goal_indices), f"Task '{task}' has {len(TASK_SPEC[task])} subgoals in TASK_SPEC but {len(goal_indices)} goals were detected"

        current_goal_idx = 0
        for i in range(episode_length):
            # Move to next goal if we've passed the current goal index
            if current_goal_idx < len(goal_indices) - 1 and i >= goal_indices[current_goal_idx]:
                current_goal_idx += 1

            target_dataset.episode_buffer["subgoal"][i] = TASK_SPEC[task][current_goal_idx]

    if "observation.points.goal_gripper_pcds" in new_features:
        for i in range(episode_length):
            current_goal_idx = target_dataset.episode_buffer["next_event_idx"][i]
            target_dataset.episode_buffer["observation.points.goal_gripper_pcds"][i] = target_dataset.episode_buffer["observation.points.gripper_pcds"][current_goal_idx[0]]

def upgrade_dataset(
    source_repo_id: str,
    target_repo_id: str,
    new_features: dict,
    remove_features: List,
    calibrations: Dict[str, Dict[str, np.ndarray]],
    discard_episodes: List[int],
    phantomize: bool,
    humanize: bool,
    path_to_extradata: Optional[str],
):
    """
    Upgrade an existing LeRobot dataset with additional features.

    Args:
        source_repo_id: Repository ID of the source dataset
        target_repo_id: Repository ID for the new dataset
        remove_features: List of features to remove from the schema
        new_features: Dictionary of new features to add to the schema
        calibrations: Dict of camera calibrations from load_calibrations()
        discard_episodes: Episodes to be discarded when upgrading
        phantomize: Source data is from human demos, upgrade to robot using output from Phantom
        humanize: Source data is from human demos, use directly without phantom retargeting
        path_to_extradata: Auxiliary data for phantomize / humanize
    """
    tolerance_s = 0.0004

    # 1. Load the existing dataset
    print(f"Loading source dataset: {source_repo_id}")
    source_dataset = LeRobotDataset(source_repo_id, tolerance_s=tolerance_s)
    source_meta = LeRobotDatasetMetadata(source_repo_id)

    # Get camera names from calibrations
    camera_names = list(calibrations.keys())

    # Get image dimensions from first camera
    first_cam = camera_names[0]
    height, width, _ = source_dataset.features[f"observation.images.{first_cam}.color"]["shape"]

    # 2. Create expanded feature schema
    print("Creating expanded feature schema...")

    # Start with existing features
    expanded_features = dict(source_dataset.features)

    # Remove any unneeded features
    for i in remove_features: expanded_features.pop(i)

    # Add new features
    expanded_features.update(new_features)

    print(f"Original features: {list(source_dataset.features.keys())}")
    print(f"New features: {list(new_features.keys())}")
    print(f"Features being removed: {remove_features}")
    print(f"Total features: {list(expanded_features.keys())}")

    # 3. Create new dataset with expanded schema
    print(f"Creating new dataset: {target_repo_id}")

    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=source_dataset.fps,
        features=expanded_features,
    )

    # 4. Upgrade data episode by episode
    print(f"Upgrading {source_meta.info['total_episodes']} episodes...")

    for episode_idx in range(source_meta.info["total_episodes"]):
        print(f"Processing episode {episode_idx + 1}/{source_meta.info['total_episodes']}")
        if episode_idx in discard_episodes:
            continue

        # Load any extra data needed for this episode
        try:
            episode_extras = _load_episode_extras(episode_idx, phantomize, humanize, path_to_extradata, camera_names)
        except AssertionError as e:
            print(f"Could not find auxiliary data for episode {episode_idx}. Skipping")
            continue

        # Get episode bounds
        episode_start = source_dataset.episode_data_index["from"][episode_idx].item()
        episode_end = source_dataset.episode_data_index["to"][episode_idx].item()
        episode_length = episode_end - episode_start

        # Process each frame in the episode
        for frame_idx in tqdm(range(episode_length)):
            original_frame = source_dataset[episode_start + frame_idx]

            frame_data = _process_frame_data(
                original_frame, source_dataset, expanded_features, source_meta,
                phantomize, humanize, episode_extras, frame_idx, episode_length,
                new_features, calibrations
            )

            target_dataset.add_frame(frame_data)

        # Process episode-level goals and events
        _process_episode_goals(
            target_dataset, episode_length, new_features, humanize,
            episode_extras, phantomize, calibrations, width, height
        )

        # Save episode
        target_dataset.save_episode()

    print(f"Upgrade complete! New dataset saved to: {target_dataset.root}")
    return target_dataset


# Example usage
if __name__ == "__main__":
    """
    Examples:

    # General command
    python upgrade_dataset.py --source_repo_id sriramsk/robot_multiview --target_repo_id sriramsk/robot_multiview_upgraded \
    --calibration_config /path/to/calibration_multiview.json --new_features gripper_pcds goal_gripper_proj calibration next_event_idx

    # Phantom retargeting mode
    python upgrade_dataset.py --source_repo_id sriramsk/human_mug_0718 --target_repo_id sriramsk/phantom_mug_0718 --discard_episodes 3 10 11 13 21 --new_features goal_gripper_proj \
    --phantomize --path_to_extradata /data/sriram/lerobot_extradata/sriramsk/human_mug_0718
    
    # Direct human data mode
    python upgrade_dataset.py --source_repo_id sriramsk/mug_on_platform_20250830_human --target_repo_id sriramsk/mug_on_platform_20250830_human_heatmapGoal --new_features gripper_pcds goal_gripper_proj next_event_idx subgoal \
    --humanize --path_to_extradata /data/sriram/lerobot_extradata/sriramsk/mug_on_platform_20250830_human
    """
    parser = argparse.ArgumentParser(description="Upgrade dataset with new keys and calibration.")
    parser.add_argument("--source_repo_id", type=str, default="sriramsk/fold_onesie_20250831_subsampled",
                        help="Source dataset repository ID")
    parser.add_argument("--target_repo_id", type=str, default="sriramsk/fold_onesie_20250831_subsampled_heatmapGoal",
                        help="Target dataset repository ID")
    parser.add_argument("--calibration_config", type=str, default="aloha_calibration/calibration_multiview.json",
                        help="Path to calibration JSON config file")
    parser.add_argument("--discard_episodes", type=int, nargs='*', default=[],
                        help="List of episode indices to discard")
    parser.add_argument("--new_features", type=str, nargs='*', default=[],
                        help="Names of new features")
    parser.add_argument("--humanize", default=False, action="store_true",
                        help="Use human data directly without phantom retargeting.")
    parser.add_argument("--phantomize", default=False, action="store_true",
                        help="Prepare new data after retargeting human data with Phantom.")
    parser.add_argument("--path_to_extradata", type=str,
                        help="Path to auxiliary data for Phantom or for human data",
                        default="/data/sriram/lerobot_extradata/")
    parser.add_argument("--push_to_hub", default=False, action="store_true",
                        help="Push upgraded dataset to HF Hub.")
    parser.add_argument("--remove_features", type=str, nargs='*', default=[],
                        help="Names of features to be removed")
    args = parser.parse_args()

    # Load calibrations
    calibrations = load_calibrations(args.calibration_config)
    camera_names = list(calibrations.keys())

    for cam_name in camera_names:
        K = calibrations[cam_name]["K"]
        scaled_K = get_scaled_intrinsics(K, (720, 1280), TARGET_SHAPE)
        calibrations[cam_name]["scaled_K"] = scaled_K
    print('scaled K calculated')

    new_features = {}
    new_features["embodiment"] = {
        'dtype': 'string',
        'shape': (1,),
        "names": ['embodiment'],
        'info': 'Name of embodiment'
    }
    if "goal_gripper_proj" in args.new_features:
        # Create goal_gripper_proj feature for each camera
        for cam_name in camera_names:
            new_features[f"observation.images.{cam_name}.goal_gripper_proj"] = {
                'dtype': 'video',
                'shape': (720, 1280, 3),
                'names': ['height', 'width', 'channels'],
                'info': 'Projection of gripper pcd at goal position onto image'
            }
    if "gripper_pcds" in args.new_features:
        new_features["observation.points.gripper_pcds"] = {
            'dtype': 'pcd',
            'shape': [-1, 3],
            'names': ['N', 'channels'],
            'info': 'Raw gripper point cloud at current position'
        }
    if "goal_gripper_pcds" in args.new_features:
        new_features["observation.points.goal_gripper_pcds"] = {
            'dtype': 'pcd',
            'shape': [-1, 3],
            'names': ['N', 'channels'],
            'info': 'Goal gripper point cloud at current position'
        }
    if "point_cloud" in args.new_features:
        new_features["observation.points.point_cloud"] = {
            'dtype': 'pcd',
            'shape': [-1, 3],
            'names': ['N', 'channels'],
            'info': 'Scene point cloud at current position'
        }
    if "next_event_idx" in args.new_features:
        new_features["next_event_idx"] = {
            'dtype': 'int32',
            'shape': (1,),
            'names': ['idx'],
            'info': 'Index of next event in the dataset'
        }
    if "subgoal" in args.new_features:
        assert "next_event_idx" in args.new_features
        new_features["subgoal"] = {
            'dtype': 'string',
            'shape': (1,),
            'names': ['subgoal'],
            'info': 'Caption for subgoal in dataset'
        }
    if "calibration" in args.new_features:
        # Add intrinsics and extrinsics for each camera
        for cam_name in camera_names:
            new_features[f"observation.{cam_name}.intrinsics"] = {
                'dtype': 'float32',
                'shape': (3, 3),
                'names': ['rows', 'cols'],
                'info': 'Camera intrinsic matrix (K)'
            }
            new_features[f"observation.{cam_name}.extrinsics"] = {
                'dtype': 'float32',
                'shape': (4, 4),
                'names': ['rows', 'cols'],
                'info': 'Camera extrinsic matrix (T_world_cam)'
            }
    remove_features = []
    if "cam_wrist" in args.remove_features:
        remove_features.append("observation.images.cam_wrist")

    assert not (args.phantomize and args.humanize), "Cannot use both phantomize and humanize modes simultaneously"
    path_to_extradata = f"{args.path_to_extradata}/{args.source_repo_id}"

    # Upgrade the dataset
    upgraded_dataset = upgrade_dataset(
        source_repo_id=args.source_repo_id,
        target_repo_id=args.target_repo_id,
        new_features=new_features,
        remove_features=remove_features,
        calibrations=calibrations,
        discard_episodes=args.discard_episodes,
        phantomize=args.phantomize,
        humanize=args.humanize,
        path_to_extradata=path_to_extradata,
    )

    if args.push_to_hub: upgraded_dataset.push_to_hub(repo_id=args.target_repo_id)

    print("Dataset upgrade completed successfully!")
    print(f"New dataset features: {list(upgraded_dataset.features.keys())}")
