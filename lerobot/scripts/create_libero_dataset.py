"""
Use data from LIBERO to create a LeRobotDataset.
"""
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.scripts.dataset_utils import generate_heatmap_from_points, project_points_to_image, get_subgoal_indices_from_gripper_actions
import torch
from tqdm import tqdm
import numpy as np
import argparse
import os
from glob import glob
import h5py
import re
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import get_real_depth_map, get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
from scipy.spatial.transform import Rotation as R

def get_4_points_from_gripper_pos_orient(gripper_pos, gripper_orn, cur_joint_angle, world_to_cam_mat=None):
    """
    From https://github.com/NakuraMino/articubot-on-mimicgen/blob/main/third_party/robogen/robogen_utils.py
    Analytically calculates 4-points on the Franka gripper

    Args:
        gripper_pos (np.ndarray): 3D position of gripper end-effector [x, y, z]
        gripper_orn (np.ndarray): Quaternion orientation of gripper [x, y, z, w]
        cur_joint_angle (float): Current gripper joint angle (0 = closed, 0.04 = open)
        world_to_cam_mat (np.ndarray, optional): 4x4 world-to-camera transform matrix.
                                               If provided, returns points in camera frame.

    Returns:
        np.ndarray: 4x3 array of gripper point cloud coordinates (world or camera frame)
    """
    original_gripper_pcd = np.array([[ 0.5648266,   0.05482348,  0.34434554],
        [ 0.5642125,   0.02702148,  0.2877661 ],
        [ 0.53906703,  0.01263776,  0.38347825],
        [ 0.54250515, -0.00441092,  0.32957944]]
    )
    original_gripper_orn = np.array([0.21120763,  0.75430543, -0.61925177, -0.05423936])

    gripper_pcd_right_finger_closed = np.array([ 0.55415434,  0.02126799,  0.32605097])
    gripper_pcd_left_finger_closed = np.array([ 0.54912525,  0.01839125,  0.3451934 ])
    gripper_pcd_closed_finger_angle = 2.6652539383870777e-05

    original_gripper_pcd[1] = gripper_pcd_right_finger_closed + (original_gripper_pcd[1] - gripper_pcd_right_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
    original_gripper_pcd[2] = gripper_pcd_left_finger_closed + (original_gripper_pcd[2] - gripper_pcd_left_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)

    goal_R = R.from_quat(gripper_orn)
    original_R = R.from_quat(original_gripper_orn)
    rotation_transfer = goal_R * original_R.inv()
    original_pcd = original_gripper_pcd - original_gripper_pcd[3]
    rotated_pcd = rotation_transfer.apply(original_pcd)
    gripper_pcd = rotated_pcd + gripper_pos

    # Transform to camera frame if transformation matrix is provided
    if world_to_cam_mat is not None:
        # Convert to homogeneous coordinates
        gripper_pcd_hom = np.hstack([gripper_pcd, np.ones((gripper_pcd.shape[0], 1))])
        # Transform to camera frame
        gripper_pcd_cam = world_to_cam_mat @ gripper_pcd_hom.T
        gripper_pcd = gripper_pcd_cam[:3].T  # Drop homogeneous coordinate

    return gripper_pcd.astype(np.float32)



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

def setup_libero_env(task_bddl_file, img_shape):
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": img_shape[0],
        "camera_widths": img_shape[1],
        "camera_depths": True,
    }
    env = OffScreenRenderEnv(**env_args)
    return env

def prepare_caption_to_bddl_mapping():
    """Map caption extracted through fname processing to a BDDL file of the environment"""
    mapping_dict = {}
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite in ["libero_goal", "libero_object", "libero_spatial", "libero_90", "libero_10"]:
        task_suite = benchmark_dict[suite]()
        for task_id in range(len(task_suite.tasks)):
            task = task_suite.get_task(task_id)
            caption = task.language
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            mapping_dict[caption] = task_bddl_file
    return mapping_dict

def gen_libero_dataset(
    repo_id: str,
    features: dict,
    file_list: list,
    img_shape: tuple,
):
    """
    Process LIBERO demonstrations to create a LeRobotDataset with multimodal features.
    
    This function converts LIBERO HDF5 demonstration files into the LeRobot dataset format,
    adding depth images, gripper point clouds, and goal projection heatmaps.

    Args:
        repo_id (str): Repository ID for the new dataset
        features (dict): Feature schema dictionary defining data types and shapes
        file_list (list): List of HDF5 file paths to process
        
    Returns:
        LeRobotDataset: Created dataset with all processed demonstrations
    """
    print(f"Creating new dataset: {repo_id}")
    CAPTION_TO_BDDL_MAPPING = prepare_caption_to_bddl_mapping()

    libero_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=features,
    )

    for h5_file in file_list:
        hf = h5py.File(h5_file)
        num_demos = len(hf['data'])

        for idx in range(num_demos):
            demo = hf[f'data/demo_{idx}']

            actions = np.asarray(demo["actions"]).astype(np.float32)
            num_steps = actions.shape[0]
            caption = get_libero_caption(h5_file)

            ee_poses = prep_ee_pose(demo).astype(np.float32)
            states = np.asarray(demo["states"])
            gripper_actions = actions[:, -1]

            # Get subgoal indices based on gripper state changes
            subgoal_indices = get_subgoal_indices_from_gripper_actions(gripper_actions)

            task_bddl_file = CAPTION_TO_BDDL_MAPPING[caption]
            env = setup_libero_env(task_bddl_file, img_shape)
            env.seed(0)
            env.reset()

            # Extract camera calibration matrices for projection computations
            agentview_ext_mat = get_camera_extrinsic_matrix(env.sim, "agentview")
            agentview_int_mat = get_camera_intrinsic_matrix(env.sim, "agentview", img_shape[1], img_shape[0])

            # The original demo doesn't contain depth so we walk through the demo
            # again in the env to render depth.
            all_obs = []
            for state in states:
                obs = env.regenerate_obs_from_state(state)
                # Convert depth to metric, store in mm
                depth = get_real_depth_map(env.sim, obs["agentview_depth"])
                obs["agentview_depth"] = (depth * 1000).astype(np.uint16)
                obs["gripper_pcd"] = get_4_points_from_gripper_pos_orient(
                    obs['robot0_eef_pos'],
                    obs['robot0_eef_quat'],
                    obs['robot0_gripper_qpos'][0],
                    np.linalg.inv(agentview_ext_mat),  # Transform to camera frame
                )
                all_obs.append(obs)

            env.close()

            for frame_idx in range(num_steps):
                frame_data = {}
                frame_data["task"] = caption
                frame_data["observation.images.agentview"] = all_obs[frame_idx]["agentview_image"]
                frame_data["observation.images.wristview"] = all_obs[frame_idx]["robot0_eye_in_hand_image"]
                frame_data["observation.images.agentview_depth"] = all_obs[frame_idx]["agentview_depth"]
                frame_data["observation.state"] = ee_poses[frame_idx]
                frame_data["action"] = actions[frame_idx]

                # Maintain the index of the next goal for each frame
                next_event_idx = next((idx for idx in subgoal_indices if idx > frame_idx), len(subgoal_indices))

                if "observation.points.gripper_pcds" in features:
                    frame_data["observation.points.gripper_pcds"] = all_obs[frame_idx]["gripper_pcd"]
                if "next_event_idx" in features:
                    frame_data["next_event_idx"] = np.array([next_event_idx], dtype=np.int32)
                if "observation.images.agentview_goal_gripper_proj" in features:
                    # Generate gripper projection heatmap for agentview camera
                    gripper_pcd_cam = all_obs[next_event_idx]["gripper_pcd"]  # Already in camera frame
                    points_2d = project_points_to_image(gripper_pcd_cam, agentview_int_mat)
                    frame_data["observation.images.agentview_goal_gripper_proj"] = generate_heatmap_from_points(points_2d, img_shape)

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
    parser.add_argument("--new_features", type=str, nargs='*', default=[],
                        help="Names of new features")
    parser.add_argument("--repo_id", type=str, default='sriramsk/libero_lerobot', help="Name of saved dataset")
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
        "observation.images.agentview": {
            'dtype': 'video',
            'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 3),
            'names': ['height', 'width', 'channels'],
            'info': 'Agentview RGB image'
        },
        "observation.images.wristview": {
            'dtype': 'video',
            'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 3),
            'names': ['height', 'width', 'channels'],
            'info': 'Wristview RGB image'
        },
        "observation.images.agentview_depth": {
            'dtype': 'video',
            'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 1),
            'names': ['height', 'width', 'channels'],
            'info': 'Agentview depth image'
        },
    }

    new_features = {}
    if "goal_gripper_proj" in args.new_features:
        new_features["observation.images.agentview_goal_gripper_proj"] = {
            'dtype': 'video',
            'shape': (IMG_SHAPE[0], IMG_SHAPE[1], 3),
            'names': ['height', 'width', 'channels'],
            'info': 'Projection of gripper pcd at goal position onto image'
        }
    if "gripper_pcds" in args.new_features:
        new_features["observation.points.gripper_pcds"] = {
            'dtype': 'float32',
            'shape': (4, 3),
            'names': ['N', 'channels'],
            'info': 'Raw gripper point cloud at current position'
        }
    if "next_event_idx" in args.new_features:
        new_features["next_event_idx"] = {
            'dtype': 'int32',
            'shape': (1,),
            'names': ['idx'],
            'info': 'Index of next event in the dataset'
        }
    features.update(new_features)

    libero_dataset = gen_libero_dataset(
        repo_id=args.repo_id,
        features=features,
        file_list=file_list,
        img_shape=IMG_SHAPE,
    )

    print("LIBERO LeRobotDataset generated successfully!")
