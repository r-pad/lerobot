from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
from tqdm import tqdm
import numpy as np
from lerobot.common.utils.aloha_utils import render_gripper_pcd
from PIL import Image
from typing import List
import argparse

def extract_events_with_gripper_pos(
    joint_states, close_thresh=15, open_thresh=25
):
    """
    First event ends when gripper closes,
    second events ends when gripper opens again.
    Only valid for 2 subgoals following this decomposition.
    """
    gripper_pos = joint_states[:, 17]
    close_gripper_idx = np.where(gripper_pos < close_thresh)[0][0]
    open_gripper_idx = close_gripper_idx + np.where(gripper_pos[close_gripper_idx:] > open_thresh)[0][0]

    return close_gripper_idx, open_gripper_idx

def get_goal_image(cam_to_world, joint_state, K, width, height, four_points=True, goal_repr="heatmap"):
    """
    Render gripper pcd in camera frame, project the full point cloud or 4 handpicked points to a mask 
    """
    mesh = render_gripper_pcd(cam_to_world=cam_to_world, joint_state=joint_state)
    assert goal_repr in ["mask", "heatmap"]

    if four_points:
        gripper_idx = np.array([6, 197, 174])
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
    
def migrate_dataset_with_new_keys(
    source_repo_id: str,
    target_repo_id: str,
    new_features: dict,
    intrinsics_txt: str,
    extrinsics_txt: str,
    discard_episodes: List[int],
):
    """
    Migrate an existing LeRobot dataset to a new one with additional features.
    
    Args:
        source_repo_id: Repository ID of the source dataset
        target_repo_id: Repository ID for the new dataset
        new_features: Dictionary of new features to add to the schema
        intrinsics_txt: Path to intrinsics txt
        extrinsics_txt: Path to extrinsics txt
        discard_episodes: Episodes to be discarded when migrating
    """
    tolerance_s = 0.0004
    cam_to_world = np.loadtxt(extrinsics_txt)
    K = np.loadtxt(intrinsics_txt)

    # 1. Load the existing dataset
    print(f"Loading source dataset: {source_repo_id}")
    source_dataset = LeRobotDataset(source_repo_id, tolerance_s=tolerance_s)
    source_meta = LeRobotDatasetMetadata(source_repo_id)

    height, width, _ = source_dataset.features["observation.images.cam_azure_kinect.color"]["shape"]

    # 2. Create expanded feature schema
    print("Creating expanded feature schema...")
    
    # Start with existing features
    expanded_features = dict(source_dataset.features)
    
    # Add new features
    expanded_features.update(new_features)
    
    print(f"Original features: {list(source_dataset.features.keys())}")
    print(f"New features: {list(new_features.keys())}")
    print(f"Total features: {list(expanded_features.keys())}")
    
    # 3. Create new dataset with expanded schema
    print(f"Creating new dataset: {target_repo_id}")
    
    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=source_dataset.fps,
        features=expanded_features,
    )

    # Define fields that LeRobot manages automatically
    AUTO_FIELDS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}

    # 4. Migrate data episode by episode
    print(f"Migrating {source_meta.info['total_episodes']} episodes...")
    
    for episode_idx in range(source_meta.info["total_episodes"]):
        print(f"Processing episode {episode_idx + 1}/{source_meta.info['total_episodes']}")
        if episode_idx in discard_episodes:
            continue

        # Get episode bounds
        episode_start = source_dataset.episode_data_index["from"][episode_idx].item()
        episode_end = source_dataset.episode_data_index["to"][episode_idx].item()
        episode_length = episode_end - episode_start

        # Process each frame in the episode
        for frame_idx in tqdm(range(episode_length)):
            # Get original frame data
            original_frame = source_dataset[episode_start + frame_idx]
            
            # Create new frame data with additional keys
            frame_data = {}
            
            # Copy existing data
            for key in source_dataset.features.keys():
                if key not in AUTO_FIELDS and key in original_frame:
                    frame_data[key] = original_frame[key]

            frame_data["task"] = source_meta.tasks[original_frame['task_index'].item()] 
            frame_data["observation.images.cam_azure_kinect.color"] = (frame_data["observation.images.cam_azure_kinect.color"].permute(1,2,0) * 255).to(torch.uint8)
            frame_data["observation.images.cam_azure_kinect.transformed_depth"] = (frame_data["observation.images.cam_azure_kinect.transformed_depth"].permute(1,2,0) * 1000).to(torch.uint16)

            # Dummy value
            frame_data["observation.images.cam_azure_kinect.goal_gripper_proj"] = torch.zeros_like(frame_data["observation.images.cam_azure_kinect.color"])

            # Add frame to new dataset
            target_dataset.add_frame(frame_data)

        joint_states = np.concatenate([target_dataset.episode_buffer['observation.state']])
        close_gripper_idx, open_gripper_idx = extract_events_with_gripper_pos(joint_states)

        goal1 = get_goal_image(cam_to_world, joint_states[close_gripper_idx], K, width, height, four_points=True)
        goal1_img = Image.fromarray(goal1).convert("RGB")
        for i in range(close_gripper_idx):
            goal1_img.save(target_dataset.episode_buffer["observation.images.cam_azure_kinect.goal_gripper_proj"][i])

        goal2 = get_goal_image(cam_to_world, joint_states[open_gripper_idx], K, width, height)
        goal2_img = Image.fromarray(goal2).convert("RGB")
        for i in range(close_gripper_idx, episode_length):
            goal2_img.save(target_dataset.episode_buffer["observation.images.cam_azure_kinect.goal_gripper_proj"][i])
        
        # Save episode
        target_dataset.save_episode()
        
    print(f"Migration complete! New dataset saved to: {target_dataset.root}")
    return target_dataset


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate dataset with new keys and optional intrinsics/extrinsics transformation.")
    parser.add_argument("--source_repo_id", type=str, default="sriramsk/aloha_mug_eef_depth",
                        help="Source dataset repository ID")
    parser.add_argument("--target_repo_id", type=str, default="sriramsk/aloha_mug_eef_depth_v3",
                        help="Target dataset repository ID")
    parser.add_argument("--intrinsics_txt", type=str, default="/home/sriram/Desktop/lerobot/lerobot/scripts/intrinsics.txt",
                        help="Path to the intrinsics.txt file")
    parser.add_argument("--extrinsics_txt", type=str, default="/home/sriram/Desktop/lerobot/lerobot/scripts/T_world_from_camera_est_v6_0709.txt",
                        help="Path to the extrinsics.txt file")
    parser.add_argument("--discard_episodes", type=int, nargs='*', default=[],
                        help="List of episode indices to discard")
    
    # Optional argument to handle new_features; update this if more structured input is needed
    parser.add_argument("--new_features", type=str, default=None,
                        help="Serialized or path to new features definition")
    args = parser.parse_args()

    new_features = {
        "observation.images.cam_azure_kinect.goal_gripper_proj": {
            'dtype': 'video', 
            'shape': (720, 1280, 3), 
            'names': ['height', 'width', 'channels'], 
            'info': 'Projection of gripper pcd at goal position onto image'}
    }

    # Migrate the dataset
    migrated_dataset = migrate_dataset_with_new_keys(
        source_repo_id=args.source_repo_id,
        target_repo_id=args.target_repo_id, 
        new_features=new_features,
        intrinsics_txt=args.intrinsics_txt,
        extrinsics_txt=args.extrinsics_txt,
        discard_episodes=args.discard_episodes
    )
    
    print("Dataset migration completed successfully!")
    print(f"New dataset features: {list(migrated_dataset.features.keys())}")
