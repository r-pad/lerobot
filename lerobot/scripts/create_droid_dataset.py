"""
Use data from DROID (original + some postprocessing) to create a LeRobotDataset.
"""
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.scripts.dataset_utils import generate_heatmap_from_points, project_points_to_image
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


def generate_heatmap_images(index, gripper_pcds, subgoal_indices, K, img_shape, GRIPPER_IDX):
    """
    gripper pointclouds are already rendered and stored in the camera frame (check r-pad/lfd3d)
    Compute heatmaps with the gripper positions at the end of each subgoal.

    Create a video of heatmaps corresponding to the goal heatmap for each frame in the rgb video.
    """
    n_imgs, height, width, _ = img_shape
    all_subgoal_heatmaps = []

    for subgoal_idx in subgoal_indices:
        subgoal_pcd = gripper_pcds[subgoal_idx][GRIPPER_IDX]
        urdf_proj = project_points_to_image(subgoal_pcd, K)
        all_subgoal_heatmaps.append(generate_heatmap_from_points(urdf_proj, (height, width)))

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

    wrist_cam_serial = metadata["wrist_cam_serial"]

    def process_camera(cam_key):
        cam_serial = metadata[f"{cam_key}_cam_serial"]
        extrinsics = metadata[f"{cam_key}_cam_extrinsics"]
        cam_to_world = get_cam_to_world(extrinsics)
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

        return cam_serial, K, world_to_cam, baseline

    cam1_serial, K1, world_to_cam1, baseline1 = process_camera("ext1")
    cam2_serial, K2, world_to_cam2, baseline2 = process_camera("ext2")

    camera_data = {
        "cam1_serial": cam1_serial,
        "cam2_serial": cam2_serial,
        "wrist_cam_serial": wrist_cam_serial,
        "cam1_K": K1,
        "cam2_K": K2,
        "world_to_cam1": world_to_cam1,
        "world_to_cam2": world_to_cam2,
        "baseline1": baseline1,
        "baseline2": baseline2,
    }

    return camera_data

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

    try:
        droid_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=15, # DROID data is recorded at 15fps but the videos have been sped up and report 60fps
            features=features,
        )
    except Exception as e:
        print("Caught exception", e)
        print("Dataset already exists? Loading existing dataset.")
        droid_dataset = LeRobotDataset(repo_id)

    num_episodes = len(dataset_indexes)
    for episode_idx in tqdm(range(droid_dataset.meta.total_episodes, num_episodes),
                            total=(num_episodes - droid_dataset.meta.total_episodes)):
        index = dataset_indexes[episode_idx]
        fname = idx_to_fname_mapping[index]
        camera_data = load_camera_params(droid_raw_dir, fname, camera_intrinsics_dict, scale_factor)

        def load_camera_data(serial, cam_type="external", K=None, baseline=None):
            # Load RGB video
            video_path = f"{droid_raw_dir}/1.0.1/{fname}/recordings/MP4/{serial}.mp4"
            images = iio.imread(video_path)
            images = np.array(
                [cv2.resize(i, (0,0), fx=scale_factor, fy=scale_factor) for i in images]
            )

            # Load depth (only for external cameras)
            depth = None
            if cam_type == "external":
                assert K is not None
                disp_name = f"{droid_raw_dir}/1.0.1/{fname}/{serial}_disp.npz"
                disparity = np.load(disp_name)["arr_0"].astype(np.float32)
                # Has already been resized
                depth = np.divide(
                    K[0, 0] * baseline,
                    disparity,
                    out=np.zeros_like(disparity),
                    where=disparity != 0,
                )
                depth = (depth * 1000.0).astype(np.uint16)[..., None]  # 16-bit in mm
            return images, depth

        # Load camera data
        cam1_images, cam1_depth = load_camera_data(
            camera_data['cam1_serial'], "external",
            camera_data['cam1_K'], camera_data['baseline1']
        )
        cam2_images, cam2_depth = load_camera_data(
            camera_data['cam2_serial'], "external",
            camera_data['cam2_K'], camera_data['baseline2']
        )
        wrist_images, _ = load_camera_data(camera_data['wrist_cam_serial'], "wrist")

        if cam1_images.shape != wrist_images.shape or cam2_images.shape != wrist_images.shape:
            print(f"Mismatched shapes between cameras. Skipping {episode_idx}")
            continue

        caption, subgoals = get_goal_text(event_dir, index)
        subgoal_indices = get_subgoal_indices(event_dir, index, subgoals)
        if subgoal_indices[-1] == 0: continue # some edge cases ...

        obs, actions = load_trajectory(droid_raw_dir, fname)

        gripper_pcds = np.load(f"{gripper_pcd_dir}/{index}.npz")["arr_0"].astype(np.float32)
        cam1_heatmap_images = generate_heatmap_images(index, gripper_pcds, subgoal_indices,
                                                      camera_data['cam1_K'], cam1_images.shape, GRIPPER_IDX)
        cam2_heatmap_images = generate_heatmap_images(index, gripper_pcds, subgoal_indices,
                                                      camera_data['cam2_K'], cam2_images.shape, GRIPPER_IDX)
        next_event_indices = np.searchsorted(subgoal_indices, np.arange(subgoal_indices[-1]), side='right')

        # Process until the end of the last subgoal
        for frame_idx in range(subgoal_indices[-1]):
            frame_data = {}
            frame_data["task"] = caption

            # Camera 1 data
            frame_data["observation.images.cam_1.color"] = cam1_images[frame_idx]
            frame_data["observation.images.cam_1.depth"] = cam1_depth[frame_idx]
            frame_data["observation.images.cam_1.goal_gripper_proj"] = cam1_heatmap_images[frame_idx]
            frame_data["observation.cam_1.intrinsics"] = camera_data['cam1_K'].astype(np.float32)
            frame_data["observation.cam_1.extrinsics"] = np.linalg.inv(camera_data['world_to_cam1']).astype(np.float32)

            # Camera 2 data
            frame_data["observation.images.cam_2.color"] = cam2_images[frame_idx]
            frame_data["observation.images.cam_2.depth"] = cam2_depth[frame_idx]
            frame_data["observation.images.cam_2.goal_gripper_proj"] = cam2_heatmap_images[frame_idx]
            frame_data["observation.cam_2.intrinsics"] = camera_data['cam2_K'].astype(np.float32)
            frame_data["observation.cam_2.extrinsics"] = np.linalg.inv(camera_data['world_to_cam2']).astype(np.float32)

            # Wrist camera
            frame_data["observation.images.cam_wrist"] = wrist_images[frame_idx]

            # Gripper point cloud
            frame_data["observation.points.gripper_pcds"] = gripper_pcds[frame_idx].astype(np.float32)

            # Next event index
            frame_data["next_event_idx"] = np.array([next_event_indices[frame_idx]], dtype=np.int32)

            # Pose and action data
            frame_data["observation.right_eef_pose"] = obs[frame_idx]
            frame_data["action.right_eef_pose"] = actions[frame_idx]
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

    Depth is available in this processed version of DROID with disparity from FoundationStereo.
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
        "observation.images.cam_wrist": {
            'dtype': 'video',
            'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 3),
            'names': ['height', 'width', 'channels'],
            'info': 'Wrist camera image'
        },
        "observation.points.gripper_pcds": {
            'dtype': 'pcd',
            'shape': [-1, 3],
            'names': ['N', 'channels'],
            'info': 'Raw gripper point cloud at current position'
        },
        "next_event_idx": {
            'dtype': 'int32',
            'shape': (1,),
            'names': ['idx'],
            'info': 'Index of next event in the dataset'
        },
    }

    for cam_name in ["cam_1", "cam_2"]:
        features[f"observation.images.{cam_name}.color"] = {
            'dtype': 'video',
            'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 3),
            'names': ['height', 'width', 'channels'],
            'info': 'RGB image'
        }
        features[f"observation.images.{cam_name}.depth"] = {
            'dtype': 'video',
            'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 1),
            'names': ['height', 'width', 'channels'],
            'info': 'Depth'
        }
        features[f"observation.images.{cam_name}.goal_gripper_proj"] = {
            'dtype': 'video',
            'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 3),
            'names': ['height', 'width', 'channels'],
            'info': 'Projection of gripper pcd at goal position onto image'
        }
        features[f"observation.{cam_name}.intrinsics"] = {
            'dtype': 'float32',
            'shape': (3, 3),
            'names': ['rows', 'cols'],
            'info': 'Camera intrinsic matrix (K)'
        }
        features[f"observation.{cam_name}.extrinsics"] = {
            'dtype': 'float32',
            'shape': (4, 4),
            'names': ['rows', 'cols'],
            'info': 'Camera extrinsic matrix (T_world_cam)'
        }

    droid_dataset = gen_droid_dataset(
        droid_path=DROID_PATH,
        repo_id="sriramsk/droid_lerobot",
        features=features,
        scale_factor=scale_factor,
    )

    print("DROID LeRobotDataset generated successfully!")
