import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import re

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
