"""
Use data from LIBERO to create a LeRobotDataset.
"""
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
from tqdm import tqdm
import numpy as np
import argparse
import os
from glob import glob
import h5py
import re

def get_libero_caption(h5_fpath):
    """
    Some hacky string processing to extract captions from the demo fname....
    """
    h5_fname = os.path.basename(h5_fpath)
    # Remove .hdf5 and _demo suffix
    base = h5_fname.replace('.hdf5', '').replace('_demo', '')
    if '_SCENE' in base:
        # Find last occurrence of SCENE[digit]_ pattern
        base = re.sub(r'^[A-Z_]+_SCENE\d+_', '', base)

    # Convert underscores to spaces
    caption = base.replace('_', ' ')
    return caption

def prep_ee_pose(demo):
    ee_states = np.asarray(demo["obs/ee_states"])
    gripper_states = np.asarray(demo["obs/gripper_states"])
    ee_poses = np.concatenate([ee_states, gripper_states], axis=1)
    return ee_poses

def gen_libero_dataset(
    repo_id: str,
    features: dict,
    file_list: list,
):
    """
    Process LIBERO to be in the LeRobotDataset format
    Args:
        repo_id: Repository ID for the new dataset
        features: Features in dataset
        file_list: List of hdf5 files to process
    """
    print(f"Creating new dataset: {repo_id}")

    try:
        libero_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=30,
            features=features,
        )
    except Exception as e:
        print("Caught exception", e)
        print("Dataset already exists? Loading existing dataset.")
        libero_dataset = LeRobotDataset(repo_id)

    for h5_file in file_list:
        hf = h5py.File(h5_file)
        num_demos = len(hf['data'])

        for idx in range(num_demos):
            demo = hf[f'data/demo_{idx}']

            actions = np.asarray(demo["actions"]).astype(np.float32)
            num_steps = actions.shape[0]
            caption = get_libero_caption(h5_file)

            agentview_imgs = np.asarray(demo["obs/agentview_rgb"])
            wristview_imgs = np.asarray(demo["obs/eye_in_hand_rgb"])
            ee_poses = prep_ee_pose(demo).astype(np.float32)

            for frame_idx in range(num_steps):
                frame_data = {}
                frame_data["task"] = caption
                frame_data["observation.images.cam.agentview"] = agentview_imgs[frame_idx]
                frame_data["observation.images.cam.wristview"] = wristview_imgs[frame_idx]
                frame_data["observation.state"] = ee_poses[frame_idx]
                frame_data["action"] = actions[frame_idx]
                libero_dataset.add_frame(frame_data)

            libero_dataset.save_episode()

    print(f"Generation complete! New dataset saved to: {libero_dataset.root}")
    return libero_dataset


if __name__ == "__main__":
    """
    python lerobot/scripts/create_libero_dataset.py --hdf5_list libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
    """
    parser = argparse.ArgumentParser(description="Generate a LeRobotDataset for LIBERO.")
    parser.add_argument("--libero_path", type=str, default="/data/sriram/libero/",
                        help="Path to LIBERO")
    parser.add_argument("--suite_names", type=str, nargs="*",
                        choices=["libero_goal", "libero_spatial", "libero_object", "libero_90", "libero_10", None],
                        help="which suite of LIBERO to process, if None set hdf5_list")
    parser.add_argument("--hdf5_list", type=str, nargs="*", default=None,
                        help="Specific HDF5 files to process")
    args = parser.parse_args()

    if (args.suite_names is None) == (args.hdf5_list is None):
        parser.error("Set only one of suite_name or hdf5_list")
    LIBERO_PATH = args.libero_path

    if args.hdf5_list is not None:
        file_list = [f"{LIBERO_PATH}/{h5_path}" for h5_path in args.hdf5_list]
    elif args.suite_names is not None:
        file_list = []
        for suite in args.suite_names:
            file_list.extend(glob(f"{LIBERO_PATH}/{suite}/*hdf5"))
    else:
        raise ValueError("No files")


    IMG_SHAPE = (128, 128)
    features = {
        "observation.state": {
            'dtype': 'float32',
            'shape': (8,),
            'names': ['ee_pos_0', 'ee_pos_1', 'ee_pos_2', 'ee_ori_0', 'ee_ori_1', 'ee_rot_2', 'gripper_state_0', 'gripper_state_1']
        },
        # NOTE: This isn't exactly delta, robosuite does some funky stuff under the hood
        # https://github.com/Lifelong-Robot-Learning/LIBERO/issues/26
        "action": {
            'dtype': 'float32',
            'shape': (7,),
            'names': ['delta_ee_pos_0', 'delta_ee_pos_1', 'delta_ee_pos_2', 'delta_ee_rot_0', 'delta_ee_rot_1', 'delta_ee_rot_2', 'gripper_action']
        },
        "observation.images.cam.agentview": {
            'dtype': 'video',
            'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 3),
            'names': ['height', 'width', 'channels'],
            'info': 'Agentview RGB image'
        },
        "observation.images.cam.wristview": {
            'dtype': 'video',
            'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 3),
            'names': ['height', 'width', 'channels'],
            'info': 'Wristview RGB image'
        },
    }

    libero_dataset = gen_libero_dataset(
        repo_id="sriramsk/libero_lerobot",
        features=features,
        file_list=file_list,
    )

    print("LIBERO LeRobotDataset generated successfully!")
