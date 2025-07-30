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
from lerobot.common.utils.aloha_utils import render_gripper_pcd
from PIL import Image
from typing import List
import argparse
import os
import imageio.v3 as iio
import pytorch3d.transforms as transforms

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

def prep_eef_pose(eef_pos, eef_rot, eef_artic):
    REAL_GRIPPER_MIN, REAL_GRIPPER_MAX = 0., 99.
    SIM_GRIPPER_MIN, SIM_GRIPPER_MAX = 0.01, 0.04
    eef_artic = (eef_artic - SIM_GRIPPER_MIN)*(
        (REAL_GRIPPER_MAX-REAL_GRIPPER_MIN)/(SIM_GRIPPER_MAX-SIM_GRIPPER_MIN)
    ) + REAL_GRIPPER_MIN
    rot_6d = transforms.matrix_to_rotation_6d(torch.from_numpy(eef_rot)).numpy()
    eef_pose = np.concatenate([rot_6d, eef_pos, eef_artic[:, None]], axis=1).astype(np.float32)
    return eef_pose

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
    
def upgrade_dataset(
    source_repo_id: str,
    target_repo_id: str,
    new_features: dict,
    intrinsics_txt: str,
    extrinsics_txt: str,
    discard_episodes: List[int],
    phantomize: bool,
    phantom_extradata: str,
):
    """
    Upgrade an existing LeRobot dataset with additional features.
    
    Args:
        source_repo_id: Repository ID of the source dataset
        target_repo_id: Repository ID for the new dataset
        new_features: Dictionary of new features to add to the schema
        intrinsics_txt: Path to intrinsics txt
        extrinsics_txt: Path to extrinsics txt
        discard_episodes: Episodes to be discarded when upgrading
        phantomize: Source data is from human demos, upgrade to robot using output from Phantom
        phantom_extradata: Auxiliary data after retargeting using Phantom
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

    # 4. Upgrade data episode by episode
    print(f"Upgrading {source_meta.info['total_episodes']} episodes...")

    PHANTOM_VID_DIR = f"{phantom_extradata}/phantom_retarget"

    for episode_idx in range(source_meta.info["total_episodes"]):
        print(f"Processing episode {episode_idx + 1}/{source_meta.info['total_episodes']}")
        if episode_idx in discard_episodes:
            continue

        if phantomize:
            vid_dir = f"{PHANTOM_VID_DIR}/episode_{episode_idx:06d}.mp4/"
            if not os.path.exists(vid_dir):
                continue
            else:
                phantom_vid = torch.from_numpy(iio.imread(f"{vid_dir}/episode_{episode_idx:06d}.mp4"))
                phantom_proprio = np.load(f"{vid_dir}/episode_{episode_idx:06d}.mp4_eef.npz")
                phantom_eef_pose = torch.from_numpy(prep_eef_pose(phantom_proprio["eef_pos"],
                                                 phantom_proprio["eef_rot"],
                                                 phantom_proprio["eef_artic"]))
                phantom_joint_state = torch.from_numpy(phantom_proprio["joint_state"]).float()

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
            if phantomize:
                rgb_data = phantom_vid[frame_idx]
                eef_data = phantom_eef_pose[frame_idx]
                joint_state = phantom_joint_state[frame_idx]
                next_idx = (frame_idx + 1) if (frame_idx + 1) < episode_length else frame_idx
                action_eef = phantom_eef_pose[next_idx]
                action = phantom_joint_state[next_idx]
            else:
                rgb_data = (frame_data["observation.images.cam_azure_kinect.color"].permute(1,2,0) * 255).to(torch.uint8)
                eef_data = frame_data["observation.right_eef_pose"]
                joint_state = frame_data["observation.state"]
                action_eef = frame_data["action.right_eef_pose"]
                action = frame_data["action"]

            frame_data["observation.images.cam_azure_kinect.color"] = rgb_data
            # TODO: Phantom should process depth as well
            frame_data["observation.images.cam_azure_kinect.transformed_depth"] = (frame_data["observation.images.cam_azure_kinect.transformed_depth"].permute(1,2,0) * 1000).to(torch.uint16)
            frame_data["observation.right_eef_pose"] = eef_data
            frame_data["observation.state"] = joint_state
            frame_data["action.right_eef_pose"] = action_eef
            frame_data["action"] = action

            if "observation.images.cam_azure_kinect.goal_gripper_proj" in new_features:
                # Dummy value
                frame_data["observation.images.cam_azure_kinect.goal_gripper_proj"] = torch.zeros_like(frame_data["observation.images.cam_azure_kinect.color"])

            # Add frame to new dataset
            target_dataset.add_frame(frame_data)

        if "observation.images.cam_azure_kinect.goal_gripper_proj" in new_features:
            joint_states = np.concatenate([target_dataset.episode_buffer['observation.state']])
            # Empirically chosen
            if phantomize: close_thresh, open_thresh = 45, 55
            else: close_thresh, open_thresh = 15, 25
            close_gripper_idx, open_gripper_idx = extract_events_with_gripper_pos(joint_states, close_thresh=close_thresh, open_thresh=open_thresh)

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
        
    print(f"Upgrade complete! New dataset saved to: {target_dataset.root}")
    return target_dataset


# Example usage
if __name__ == "__main__":
    """
    python upgrade_dataset.py --source_repo_id sriramsk/human_mug_0718 --target_repo_id sriramsk/phantom_mug_0718 --discard_episodes 3 10 11 13 21 --new_features goal_gripper_proj \
    --phantomize --phantom_extradata /data/sriram/lerobot_extradata/sriramsk/human_mug_0718
    """
    parser = argparse.ArgumentParser(description="Upgrade dataset with new keys and optional intrinsics/extrinsics transformation.")
    parser.add_argument("--source_repo_id", type=str, default="sriramsk/human_mug_0718",
                        help="Source dataset repository ID")
    parser.add_argument("--target_repo_id", type=str, default="sriramsk/phantom_mug_0718",
                        help="Target dataset repository ID")
    parser.add_argument("--intrinsics_txt", type=str, default="/home/sriram/Desktop/lerobot/lerobot/scripts/intrinsics.txt",
                        help="Path to the intrinsics.txt file")
    parser.add_argument("--extrinsics_txt", type=str, default="/home/sriram/Desktop/lerobot/lerobot/scripts/T_world_from_camera_est_v6_0709.txt",
                        help="Path to the extrinsics.txt file")
    parser.add_argument("--discard_episodes", type=int, nargs='*', default=[],
                        help="List of episode indices to discard")
    parser.add_argument("--new_features", type=str, nargs='*', default=[],
                        help="Names of new features")
    parser.add_argument("--phantomize", default=False, action="store_true",
                        help="Prepare new data after retargeting using Phantom.")
    parser.add_argument("--phantom_extradata", type=str,
                        help="Path to auxiliary Phantom data")
    parser.add_argument("--push_to_hub", default=False, action="store_true",
                        help="Push upgraded dataset to HF Hub.")
    args = parser.parse_args()

    new_features = {}
    if "goal_gripper_proj" in args.new_features:
        new_features["observation.images.cam_azure_kinect.goal_gripper_proj"] = {
            'dtype': 'video',
            'shape': (720, 1280, 3),
            'names': ['height', 'width', 'channels'],
            'info': 'Projection of gripper pcd at goal position onto image'
        }
    assert (args.phantom_extradata is not None) if args.phantomize else True
    phantom_extradata = args.phantom_extradata if args.phantomize else None

    # Upgrade the dataset
    upgraded_dataset = upgrade_dataset(
        source_repo_id=args.source_repo_id,
        target_repo_id=args.target_repo_id, 
        new_features=new_features,
        intrinsics_txt=args.intrinsics_txt,
        extrinsics_txt=args.extrinsics_txt,
        discard_episodes=args.discard_episodes,
        phantomize=args.phantomize,
        phantom_extradata=args.phantom_extradata,
    )

    if args.push_to_hub: upgraded_dataset.push_to_hub(repo_id=args.target_repo_id)
    
    print("Dataset upgrade completed successfully!")
    print(f"New dataset features: {list(upgraded_dataset.features.keys())}")
