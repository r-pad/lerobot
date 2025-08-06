"""
Use data from DROID (original + some postprocessing) to create a LeRobotDataset.
"""
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from typing import List
import argparse
import os
import json
from glob import glob
import cv2
import imageio.v3 as iio
import imageio
import h5py
from scipy.spatial.transform import Rotation
import pytorch3d.transforms as transforms


def generate_heatmap_images(index, gripper_pcd_dir, subgoal_indices, K, img_shape, GRIPPER_IDX):
    """
    gripper pointclouds are already rendered and stored in the camera frame (check r-pad/lfd3d)
    Compute heatmaps with the gripper positions at the end of each subgoal.

    Create a video of heatmaps corresponding to the goal heatmap for each frame in the rgb video.
    """
    n_imgs, height, width, _ = img_shape
    gripper_pcds = np.load(f"{gripper_pcd_dir}/{index}.npz")["arr_0"].astype(
        np.float32
    )
    all_subgoal_heatmaps = []

    def compute_heatmap(urdf_proj, width, height):
        goal_image = np.zeros((height, width, 3))
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

    for subgoal_idx in subgoal_indices:
        subgoal_pcd = gripper_pcds[subgoal_idx][GRIPPER_IDX]
        proj_hom = (K @ subgoal_pcd.T).T
        urdf_proj = (proj_hom / proj_hom[:, 2:])[:, :2]
        urdf_proj = np.clip(urdf_proj, [0, 0], [width - 1, height - 1]).astype(int)
        all_subgoal_heatmaps.append(compute_heatmap(urdf_proj, width, height))

    image_indices = np.searchsorted(subgoal_indices, np.arange(subgoal_indices[-1]), side='right')
    heatmap_images = np.array([all_subgoal_heatmaps[idx] for idx in image_indices])
    return heatmap_images

def load_trajectory(droid_raw_dir, fname):
    """
    Load obs/action data in the format we're currently using with the Aloha.
    """
    trajectory = h5py.File(f"{droid_raw_dir}/1.0.1/{fname}/trajectory.h5")

    def prep_eef_pose(trajectory, key):
        gripper_cartesian_pos = np.array(
                trajectory[f"{key}/robot_state/cartesian_position"]
            )
        gripper_action = np.array(
                trajectory[f"{key}/robot_state/gripper_position"]
            )
        ee_pos = gripper_cartesian_pos[:, :3]
        ee_euler = gripper_cartesian_pos[:, 3:6]

        R_ee = Rotation.from_euler("xyz", ee_euler)
        # Align the coordinate frames in DROID with the one in the Aloha
        R_aloha_align = Rotation.from_euler("xy", [180, -90], degrees=True)
        R_ee = R_ee * R_aloha_align
        R_ee = R_ee.as_matrix()

        ee_rot = transforms.matrix_to_rotation_6d(torch.from_numpy(R_ee)).numpy()
        eef_pose = np.concatenate([ee_rot, ee_pos, gripper_action[:, None]], axis=1).astype(np.float32)
        return eef_pose

    obs = prep_eef_pose(trajectory, "observation")
    action = prep_eef_pose(trajectory, "action")

    return obs, action

def parse_timestamp(timestamp):
    """Convert MM:SS format to seconds"""
    minutes, seconds = map(int, timestamp.split(":"))
    return minutes * 60 + seconds

def get_subgoal_indices(event_dir, index, subgoals):
    """
    Event videos were preprocessed to be of length 20s before feeding
    to Gemini. So they have varying fps - get frame index of subgoal
    by checking fps of this preprocessed video.

    Return list of frame indices where a subgoal ends.
    """
    video_path = f"{event_dir}/{index}/video.mp4"
    with imageio.get_reader(video_path) as reader:
        fps = reader.get_meta_data()['fps']

    subgoal_indices = []
    for subgoal_idx in range(len(subgoals)):
        end_time = subgoals[subgoal_idx]["timestamp"]
        # Convert timestamps to frame indices
        end_frame_idx = int(parse_timestamp(end_time) * fps)
        subgoal_indices.append(end_frame_idx)
    return subgoal_indices

def load_gripper_pcd(index, gripper_pcd_dir, event_start_idx, event_end_idx):
    gripper_pcds = np.load(f"{gripper_pcd_dir}/{index}.npz")["arr_0"].astype(
        np.float32
    )
    start_tracks = gripper_pcds[event_start_idx]
    end_tracks = gripper_pcds[event_end_idx]
    return start_tracks, end_tracks

def get_goal_text(event_dir, index):
    """
    Concatenate all the subgoals from Gemini into one big caption.
    """
    with open(f"{event_dir}/{index}/subgoal.json") as f:
        subgoals = json.load(f)
    return " and ".join(d["subgoal"] for d in subgoals), subgoals


def get_cam_to_world(cam_extrinsics):
    rotation = Rotation.from_euler("xyz", np.array(cam_extrinsics[3:])).as_matrix()
    translation = cam_extrinsics[:3]

    cam_to_world = np.zeros((4, 4))
    cam_to_world[:3, :3] = rotation
    cam_to_world[:3, 3] = translation
    cam_to_world[3, 3] = 1
    return cam_to_world

def load_camera_params(droid_raw_dir, fname, camera_intrinsics_dict, scale_factor):
    metadata_file = glob(f"{droid_raw_dir}/1.0.1/{fname}/metadata*json")[0]
    with open(metadata_file) as f:
        metadata = json.load(f)

    # We work with camera-1 data across the dataset.
    cam_serial = metadata["ext1_cam_serial"]
    extrinsics_left = metadata["ext1_cam_extrinsics"]
    cam_to_world = get_cam_to_world(extrinsics_left)
    world_to_cam = np.linalg.inv(cam_to_world)

    cam_params = camera_intrinsics_dict[cam_serial]
    K = np.array(
        [
            [cam_params["fx"], 0.0, cam_params["cx"]],
            [0.0, cam_params["fy"], cam_params["cy"]],
            [0.0, 0.0, 1.0],
        ]
    )
    K = K * scale_factor
    K[2, 2] = 1
    baseline = cam_params["baseline"] / 1000
    return cam_serial, K, baseline, world_to_cam

def gen_droid_dataset(
    droid_path: str,
    repo_id: str,
    features: dict,
    scale_factor: float,
):
    """
    Process DROID to be in the LeRobotDataset format with some extra keys
    from auxiliary info.

    Args:
        droid_path: Path to DROID + auxiliary info on disk
        repo_id: Repository ID for the new dataset
        features: Features in dataset
        scale_factor: Scaling factor
    """
    droid_raw_dir = f"{droid_path}/droid_raw"  # raw videos, depth and metadata
    gripper_pcd_dir = f"{droid_path}/droid_gripper_pcd"  # gripper pcd rendered from Mujoco - see r-pad/lfd3d
    event_dir = f"{droid_path}/droid_gemini_events"  # Subgoals and videos - see r-pad/lfd3d
    # indexes of selected gripper points -> handpicked
    GRIPPER_IDX = np.array([356, 232, 16])

    # Intrinsics of all cameras in DROID
    with open("droid_meta/zed_intrinsics.json") as f:
        camera_intrinsics_dict = json.load(f)
    # From some postprocessing on the DROID dataset to extract a mapping of index in DROID to its filepath in DROID-raw
    with open("droid_meta/idx_to_fname_mapping.json") as f:
        idx_to_fname_mapping = json.load(f)
    # From https://medium.com/@zubair_irshad/scaling-up-automatic-camera-calibration-for-droid-dataset-4ddfc45361d3
    with open("droid_meta/all_valid_indexes.json") as f:
        dataset_indexes = json.load(f)

    print(f"Creating new dataset: {repo_id}")

    droid_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=60, # DROID data is recorded at 60fps
        features=features,
    )

    num_episodes = len(dataset_indexes)
    for episode_idx in tqdm(range(num_episodes), total=num_episodes):
        index = dataset_indexes[episode_idx]
        fname = idx_to_fname_mapping[index]
        cam_serial, K, baseline, world_to_cam = load_camera_params(droid_raw_dir, fname, camera_intrinsics_dict, scale_factor)

        video_path = f"{droid_raw_dir}/1.0.1/{fname}/recordings/MP4/{cam_serial}.mp4"
        images = iio.imread(video_path)
        images = np.array(
            [cv2.resize(i, (0,0), fx=scale_factor, fy=scale_factor) for i in images]
        )

        caption, subgoals = get_goal_text(event_dir, index)
        subgoal_indices = get_subgoal_indices(event_dir, index, subgoals)
        obs, actions = load_trajectory(droid_raw_dir, fname)
        heatmap_images = generate_heatmap_images(index, gripper_pcd_dir, subgoal_indices,
                                                 K, images.shape, GRIPPER_IDX)

        # Process until the end of the last subgoal
        for frame_idx in range(subgoal_indices[-1]):
            frame_data = {}
            frame_data["task"] = caption
            frame_data["observation.images.cam.color"] = images[frame_idx]
            frame_data["observation.right_eef_pose"] = obs[frame_idx]
            frame_data["action.right_eef_pose"] = actions[frame_idx]
            frame_data["observation.images.cam.goal_gripper_proj"] = heatmap_images[frame_idx]
            frame_data["action"] = np.zeros(18, dtype=np.float32)
            frame_data["observation.state"] = np.zeros(18, dtype=np.float32)

            droid_dataset.add_frame(frame_data)

        droid_dataset.save_episode()

    print(f"Generation complete! New dataset saved to: {droid_dataset.root}")
    return droid_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a LeRobotDataset for DROID.")
    parser.add_argument("--droid_path", type=str, default="/data/sriram/autobot_mount/DROID/",
                        help="Path to DROID and postprocessed data")
    args = parser.parse_args()
    DROID_PATH = args.droid_path

    # Original video from DROID is (720, 1280) its downsampled 4x to save space
    ORIG_SHAPE = (720, 1280)
    scale_factor = 0.25
    IMG_SHAPE = (int(ORIG_SHAPE[0] * scale_factor), int(ORIG_SHAPE[1] * scale_factor))

    """
    NOTE: Some info about states/actions:
    observation.state and action are currently placeholders with keys and shapes copied from the Aloha
    This data is available in DROID and can probably be saved but its not really useful for cross-embodiment training anyway ...

    `right_eef_pose` is again misleading because there's no right/left arm in DROID but this is just following Aloha conventions
    since existing code has the right_eef_pose key. Should probably rename both keys to just be eef_pose.

    Depth is available in this processed version of DROID with disparity from FoundationStereo, currently not saved.
    """
    features = {
        "observation.state": {
            'dtype': 'float32',
            'shape': (18,),
            'names': ['left_waist', 'left_shoulder', 'left_shoulder_shadow', 'left_elbow', 'left_elbow_shadow', 'left_forearm_roll', 'left_wrist_angle', 'left_wrist_rotate', 'left_gripper', 'right_waist', 'right_shoulder', 'right_shoulder_shadow', 'right_elbow', 'right_elbow_shadow', 'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate', 'right_gripper']
        },
        "observation.right_eef_pose": {
            'dtype': 'float32',
            'shape': (10,),
            'names': ['rot_6d_0', 'rot_6d_1', 'rot_6d_2', 'rot_6d_3', 'rot_6d_4', 'rot_6d_5', 'trans_0', 'trans_1', 'trans_2', 'gripper_articulation']
        },
        "action": {
            'dtype': 'float32',
            'shape': (18,),
            'names': ['left_waist', 'left_shoulder', 'left_shoulder_shadow', 'left_elbow', 'left_elbow_shadow', 'left_forearm_roll', 'left_wrist_angle', 'left_wrist_rotate', 'left_gripper', 'right_waist', 'right_shoulder', 'right_shoulder_shadow', 'right_elbow', 'right_elbow_shadow', 'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate', 'right_gripper']
        },
        "action.right_eef_pose": {
            'dtype': 'float32',
            'shape': (10,),
            'names': ['rot_6d_0', 'rot_6d_1', 'rot_6d_2', 'rot_6d_3', 'rot_6d_4', 'rot_6d_5', 'trans_0', 'trans_1', 'trans_2', 'gripper_articulation']
        },
        # "observation.images.cam.depth": {
        #     'dtype': 'video',
        #     'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 1),
        #     'names': ['height', 'width', 'channels'],
        #     'info': 'Depth'
        # },
        "observation.images.cam.color": {
            'dtype': 'video',
            'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 3),
            'names': ['height', 'width', 'channels'],
            'info': 'RGB image'
        },
        "observation.images.cam.goal_gripper_proj": {
            'dtype': 'video',
            'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 3),
            'names': ['height', 'width', 'channels'],
            'info': 'Projection of gripper pcd at goal position onto image'
        },
    }

    droid_dataset = gen_droid_dataset(
        droid_path=DROID_PATH,
        repo_id="sriramsk/droid_lerobot",
        features=features,
        scale_factor=scale_factor,
    )

    print("DROID LeRobotDataset generated successfully!")
