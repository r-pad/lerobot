
from pprint import pprint

import torch
from huggingface_hub import HfApi
import os
import pickle
import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from PIL import Image
from pathlib import Path
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix   
from pytorch3d.ops import sample_farthest_points
import time
from torchvision import transforms
import torch.nn.functional as F
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from termcolor import cprint

### TODO: add torch transformations for resizing the rgb and depth images
target_shape = 224
rgb_preprocess = transforms.Compose(
    [
        transforms.Resize(
            target_shape,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(target_shape),
    ]
)
depth_preprocess = transforms.Compose(
    [
        transforms.Resize(
            target_shape,
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.CenterCrop(target_shape),
    ]
)

def get_scaled_intrinsics(K, orig_shape, target_shape=224):
    """
    Scale camera intrinsics matrix based on image resizing and cropping.

    Args:
        K (np.ndarray): Original camera intrinsics matrix (3x3).
        orig_shape (tuple): Original image shape (height, width).
        target_shape (int): Target size for resized images (default: 224).

    Returns:
        np.ndarray: Scaled intrinsics matrix (3x3).
    """
    # Getting scale factor from torchvision.transforms.Resize behaviour
    K_ = K.copy()
    scale_factor = target_shape / min(orig_shape)

    # Apply the scale factor to the intrinsics
    K_[[0, 1], [0, 1]] *= scale_factor  # fx, fy
    K_[[0, 1], 2] *= scale_factor  # cx, cy

    # Adjust the principal point (cx, cy) for the center crop
    crop_offset_x = (orig_shape[1] * scale_factor - target_shape) / 2
    crop_offset_y = (orig_shape[0] * scale_factor - target_shape) / 2
    K_[0, 2] -= crop_offset_x
    K_[1, 2] -= crop_offset_y
    return K_


### NOTE: load a dino-v2 model
dinov2 = torch.hub.load(
    "facebookresearch/dinov2", "dinov2_vitl14_reg"
).to("cuda")
def get_dinov2_image_embedding(image, dinov2=None, device="cuda"):
    if dinov2 is None:
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(
            device
        )
    patch_size = 14
    target_shape = 224

    assert type(image) == Image.Image
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                target_shape, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(target_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    inputs = preprocess(image).unsqueeze(0).to(device)

    # Forward pass to get features
    with torch.no_grad():
        outputs = dinov2.forward_features(inputs)

    # Extract the last hidden state as features
    patch_features = outputs["x_norm_patchtokens"].squeeze(0)
    num_patches = patch_features.shape[0]
    h = w = int(num_patches**0.5)
    patch_features_2d = patch_features.reshape(h, w, -1)

    # Permute to [C, H, W] for interpolation
    patch_features_2d = patch_features_2d.permute(2, 0, 1)

    # Upsample to match original image patch dimensions
    resized_features = F.interpolate(
        patch_features_2d.unsqueeze(0),
        size=(target_shape, target_shape),
        mode="bilinear",
        align_corners=False,
    )

    return resized_features.squeeze().permute(1, 2, 0).cpu().numpy()

def compute_dino_v2_features(rgb, target_shape=224):
    # pca_n_components = 256
    rgb_embed = get_dinov2_image_embedding(
        Image.fromarray(rgb), dinov2=dinov2, device="cuda"
    )

    # pca_model = PCA(n_components=pca_n_components)
    # rgb_embed = pca_model.fit_transform(
    #     rgb_embed.reshape(-1, rgb_embed.shape[2])
    # # )
    # rgb_embed = rgb_embed.reshape(
    #     target_shape, target_shape, -1
    # )
    
    return rgb_embed.reshape(-1, rgb_embed.shape[2])  # (H*W, feat_dim)

robot_base_in_table_center_frame = np.array([0.449, -0.019, 0.00])
R_world_to_robot = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]]
)

def rotation_transfer_6D_to_matrix(rot_6d):
    return rotation_6d_to_matrix(torch.from_numpy(rot_6d)[None, :]).squeeze().cpu().numpy()

def rotation_transfer_matrix_to_6D(rot_matrix):
    return matrix_to_rotation_6d(torch.from_numpy(rot_matrix)[None, :]).squeeze().cpu().numpy()

def R_y(deg):
    """Rotation matrix for rotation about +y by 'deg' degrees."""
    theta = np.deg2rad(deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c],
    ])

def R_z(deg):
    theta = np.deg2rad(deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, s,  0],
        [ -s, c,  0],
        [0, 0,  1],
    ])


# Fixed transform: viper frame → franka frame
# R_fv = R_y(-90.0) 
R_vf_y = R_y(90.0)
R_vf_z = R_z(180)

def transform_from_table_center_to_robot_base(xyz_in_table_center):
    # xyz_robot_base = xyz_in_table_center - robot_base_in_table_center_frame
    # xyz_robot_base[:, 0] = -xyz_robot_base[:, 0]
    T_robot_to_table_world = np.array([
        [-1, 0, 0, 0.449],
        [0, -1, 0, -0.019,],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T_table_to_robot = np.linalg.inv(T_robot_to_table_world)
    N = xyz_in_table_center.shape[0]
    homo = np.ones((N, 1))
    xyz_in_table_center_homo = np.concatenate([xyz_in_table_center, homo], axis=1) # N x 4
    xyz_in_robot_homo = (T_table_to_robot @ xyz_in_table_center_homo.T).T
    xyz_in_robot = xyz_in_robot_homo[:, :3]
    return xyz_in_robot

def get_4_points_from_gripper_pos_orient(gripper_pos, gripper_orn, cur_joint_angle):
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
 
    # goal_R = R.from_quat(gripper_orn)
    # import pdb; pdb.set_trace()
    goal_R = R.from_matrix(gripper_orn)
    original_R = R.from_quat(original_gripper_orn)
    rotation_transfer = goal_R * original_R.inv()
    original_pcd = original_gripper_pcd - original_gripper_pcd[3]
    rotated_pcd = rotation_transfer.apply(original_pcd)
    gripper_pcd = rotated_pcd + gripper_pos
    return gripper_pcd

def get_fps_pcd(scene_pcd, num_points):
    scene_pcd_downsample, scene_points_idx = sample_farthest_points(
        scene_pcd, K=num_points, random_start_point=False
    )
    scene_pcd = scene_pcd_downsample.squeeze().cpu().numpy()  # (num_points, 3)
    return scene_pcd

def fpsample_pcd(scene_pcd, num_points):
    import fpsample
    h = min(9, np.log2(num_points))
    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(scene_pcd[:, :3], num_points, h=h)
    kdline_fps_samples_idx = np.array(sorted(kdline_fps_samples_idx))
    scene_pcd = scene_pcd[kdline_fps_samples_idx]
    return scene_pcd, kdline_fps_samples_idx


def get_scene_pcd_cam_frame(depth, K, num_points, max_depth=None):
    """
    Generate a downsampled point cloud (PCD) from RGB embeddings and depth map.

    Args:
        rgb_embed (np.ndarray): RGB feature embeddings of shape (H, W, feat_dim).
        depth (np.ndarray): Depth map of shape (H, W).
        K (np.ndarray): Camera intrinsics matrix (3x3).
        num_points (int): Number of points to sample from the PCD.
        max_depth (float): Maximum depth value for valid points.

    Returns:
        tuple: (scene_pcd, scene_feat_pcd) where:
            - scene_pcd (np.ndarray): Downsampled 3D points of shape (num_points, 3).
            - scene_feat_pcd (np.ndarray): Features for downsampled points of shape (num_points, feat_dim).
    """
    height, width = depth.shape
    # Create pixel coordinate grid
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    x_flat, y_flat, z_flat = x_grid.flatten(), y_grid.flatten(), depth.flatten()

    # Remove points with invalid depth
    valid_depth = np.logical_and(z_flat > 0, z_flat < max_depth)

    x_flat, y_flat, z_flat = (
        arr[valid_depth] for arr in (x_flat, y_flat, z_flat)
    )
    # Unproject points using K inverse
    pixels = np.stack([x_flat, y_flat, np.ones_like(x_flat)], axis=0)
    K_inv = np.linalg.inv(K)
    points = (K_inv @ pixels) * z_flat  # Shape: (3, N)
    points = points.T  # Shape: (N, 3)

    scene_pcd = torch.from_numpy(points[None])  # (1, N, 3)

    if num_points is not None:
        scene_pcd = get_fps_pcd(
            scene_pcd, num_points
        )
    else:
        scene_pcd = scene_pcd[0]

    return scene_pcd, valid_depth

def _load_camera_intrinsics(intrinsics_path):
    """Load camera intrinsics from file.

    Args:
        intrinsics_path: Relative path to intrinsics file (e.g., "aloha_calibration/intrinsics_xxx.txt")

    Returns:
        np.ndarray: 3x3 intrinsics matrix
    """
    file_path = intrinsics_path
    return np.loadtxt(file_path)

def _load_camera_extrinsics(extrinsics_path):
    """Load camera extrinsics (T_world_from_camera) from file.

    Args:
        extrinsics_path: Relative path to extrinsics file (e.g., "aloha_calibration/T_world_from_camera_xxx.txt")

    Returns:
        np.ndarray: 4x4 transformation matrix (T_world_from_camera)
    """
    file_path = extrinsics_path
    T = np.loadtxt(file_path).astype(np.float32)
    return T.reshape(4, 4)

def transform_to_world_frame(points_cam, T_world_from_cam):
    """Transform points from camera frame to world frame.

    Args:
        points_cam: (N, 3) array of points in camera frame
        T_world_from_cam: (4, 4) transformation matrix

    Returns:
        (N, 3) array of points in world frame
    """
    # Convert to homogeneous coordinates
    N = points_cam.shape[0]
    points_hom = np.concatenate([points_cam, np.ones((N, 1))], axis=1)  # (N, 4)

    # Apply transformation
    points_world_hom = (T_world_from_cam @ points_hom.T).T  # (N, 4)

    # Convert back to 3D
    points_world = points_world_hom[:, :3]  # (N, 3)

    return points_world

def get_gripper_4_points_from_sriram_data(eef_pose_from_sriram):
    right_eef_pose = eef_pose_from_sriram
    eef_rot_6d = right_eef_pose[:6]
    eef_rot_matrix = rotation_6d_to_matrix(eef_rot_6d[None, :]).squeeze().cpu().numpy()  # (3, 3)
    # eef_rot_matrix_franka_eef_coordinate = R_fv @ eef_rot_matrix
    eef_rot_matrix_franka_eef_coordinate = eef_rot_matrix @ R_vf_y @ R_vf_z
    eef_rot_matrix_robot_base = R_world_to_robot @ eef_rot_matrix_franka_eef_coordinate
    # eef_rot_matrix_robot_base = eef_rot_matrix # @ R_world_to_robot 
    eef_rot_6d_robot_base = matrix_to_rotation_6d(torch.from_numpy(eef_rot_matrix_robot_base)[None, :]).squeeze().cpu().numpy()
    
    eef_pos = right_eef_pose[6:9].cpu().numpy()
    eef_pos_robot_base = transform_from_table_center_to_robot_base(eef_pos[None, :]).squeeze()
    
    ### NOTE: perform the conversion of the aloha eef gripper width to panda gripper width
    eef_gripper_width = right_eef_pose[9:10].cpu().numpy()
    eef_gripper_width_franka = eef_gripper_width / 80 * 0.04  # convert to franka gripper width
    eef_gripper_width_franka = np.clip(eef_gripper_width_franka, 0, 0.04)

    return eef_pos, eef_rot_6d, eef_gripper_width, eef_pos_robot_base, eef_rot_matrix_robot_base, eef_rot_6d_robot_base, eef_gripper_width_franka
    

def extract_actions(traj_pos_ori, traj_pc, traj_gripper_pcd, traj_goal_gripper_pcd, traj_rgb_values, traj_rgb_features, combine_action_steps=2):
    traj_pos_ori = traj_pos_ori[::combine_action_steps]
    traj_pc = traj_pc[::combine_action_steps]
    traj_gripper_pcd = traj_gripper_pcd[::combine_action_steps]
    traj_goal_gripper_pcd = traj_goal_gripper_pcd[::combine_action_steps]
    if traj_rgb_features is not None:
        traj_rgb_features = traj_rgb_features[::combine_action_steps]
    traj_rgb_values = traj_rgb_values[::combine_action_steps]
    
    traj_actions = []
    
    filtered_pcs = []
    filtered_pos_oris = []
    filtered_gripper_pcds = []
    filtered_goal_gripper_pcds = []
    filtered_rgb_features = []
    filtered_rgb_values = []
    
    base_pos = traj_pos_ori[0][:3]
    base_ori_6d = traj_pos_ori[0][3:9]
    base_finger_angle = traj_pos_ori[0][9]
    base_gripper_pcd = traj_gripper_pcd[0]
    base_pc = traj_pc[0]
    base_pos_ori = traj_pos_ori[0]
    base_goal_gripper_pcd = traj_goal_gripper_pcd[0]
    if traj_rgb_features is not None:
        base_rgb_features = traj_rgb_features[0]
    base_rgb_values = traj_rgb_values[0]
    
    beg = time.time()
    for i in range(len(traj_pos_ori) - 1):
        target_pos = traj_pos_ori[i+1][:3]

        ### translation
        delta_pos = np.array(target_pos) - np.array(base_pos)
        target_ori_6d = traj_pos_ori[i+1][3:9]

        ### rotation
        base_ori_matrix = rotation_transfer_6D_to_matrix(base_ori_6d)
        target_ori_matrix = rotation_transfer_6D_to_matrix(target_ori_6d)
        delta_ori_matrix = base_ori_matrix.T @ target_ori_matrix
        delta_ori_6d = rotation_transfer_matrix_to_6D(delta_ori_matrix)
                                        
        target_finger_angle = traj_pos_ori[i+1][9]
        delta_finger_angle = target_finger_angle - base_finger_angle
                
        ### make sure we learn the delta finger open close with a pretty large value
        delta_finger_angle_threshold = 1 / 80 * 0.04
        if np.abs(delta_finger_angle) > delta_finger_angle_threshold:
            # print("setting large finger change at time step ", i)
            delta_finger_angle = np.sign(delta_finger_angle) * 0.006
        
        filter_action = False
        if filter_action:
            continue
        else:
            action = delta_pos.tolist() + delta_ori_6d.tolist() + [delta_finger_angle]
            traj_actions.append(action)
            filtered_pcs.append(base_pc)
            filtered_gripper_pcds.append(base_gripper_pcd)
            filtered_pos_oris.append(base_pos_ori)
            filtered_goal_gripper_pcds.append(base_goal_gripper_pcd)
            if traj_rgb_features is not None:
                filtered_rgb_features.append(base_rgb_features)
            filtered_rgb_values.append(base_rgb_values)
            
            base_pc = traj_pc[i+1]
            base_gripper_pcd = traj_gripper_pcd[i+1]
            base_pos_ori = traj_pos_ori[i+1]
            base_pos = target_pos
            base_ori_6d = target_ori_6d
            base_finger_angle = target_finger_angle
            base_goal_gripper_pcd = traj_goal_gripper_pcd[i+1]
            if traj_rgb_features is not None:
                base_rgb_features = traj_rgb_features[i+1]
            base_rgb_values = traj_rgb_values[i+1]
    cprint("extract actions using time: {}".format(time.time() - beg), "red")

    return np.asarray(traj_actions), filtered_pcs, filtered_pos_oris, filtered_gripper_pcds, filtered_goal_gripper_pcds, filtered_rgb_values, filtered_rgb_features


def process_one_frame(
    dataset,
    dataset_idx: int,
    traj_idx: int,
    from_idx: int,
    all_intrinsics,
    all_extrinsics,
    max_depth: float,
    num_points: int,
    store_dino: bool,
    target_shape: int,
):
    """
    Returns everything you used to append inside the loop, but for one frame.
    IMPORTANT: no shared writes here.
    """
    data_point = dataset[dataset_idx]

    # sanity checks (optional, but helps catch mismatches early)
    # episode_index = data_point["episode_index"].item()
    # frame_index = data_point["frame_index"].item()
    # assert episode_index == traj_idx
    # assert frame_index == (dataset_idx - from_idx)
    
    if dataset_idx % 100 == 0:
        print(f"Processing traj {traj_idx}, dataset idx {dataset_idx}")

    wrist_image = data_point["observation.images.cam_wrist"].permute(1, 2, 0).cpu().numpy()
    rgb_image_front = data_point["observation.images.cam_azure_kinect_front.color"].permute(1, 2, 0).cpu().numpy()
    rgb_image_back = data_point["observation.images.cam_azure_kinect_back.color"].permute(1, 2, 0).cpu().numpy()

    return {
        "wrist_image": wrist_image,
        "rgb_image_front": rgb_image_front,
        "rgb_image_back": rgb_image_back,
    }
    
def _save_step_npz(
    t_idx: int,
    traj_dir: str,
    traj_wrist_images,
    traj_rgb_images_front,
    traj_rgb_images_back,
):
    step_save_path = os.path.join(traj_dir, f"{t_idx}.npz")
    ### first read all the data from step_save_path
    existing_data = np.load(step_save_path, allow_pickle=True)
    traj_eef_pose = existing_data["state"]
    traj_scene_pcd = existing_data["point_cloud"]
    action_arrays = existing_data["action"]
    traj_gripper_pcd = existing_data["gripper_pcd"]
    traj_goal_gripper_pcd = existing_data["goal_gripper_pcd"]
    traj_fpsed_rgb_values = existing_data["rgb_values"]
    if "rgb_features" in existing_data:
        traj_fpsed_rgb_features = existing_data["rgb_features"]
    else:
        traj_fpsed_rgb_features = None

    data = {
        "state": traj_eef_pose,
        "point_cloud": traj_scene_pcd,
        "action": action_arrays,
        "gripper_pcd": traj_gripper_pcd,
        "goal_gripper_pcd": traj_goal_gripper_pcd,
        "rgb_values": (traj_fpsed_rgb_values.astype(np.float32) / 255.0),
    }
    if store_dino:
        data["rgb_features"] = traj_fpsed_rgb_features
        
    data['wrist_image'] = traj_wrist_images[t_idx][None, :, :, :]
    data['rgb_image_front'] = traj_rgb_images_front[t_idx][None, :, :, :]
    data['rgb_image_back'] = traj_rgb_images_back[t_idx][None, :, :, :]

    # Atomic-ish write: write to temp then rename (optional but safer)
    tmp_path = step_save_path # + ".tmp"
    np.savez_compressed(tmp_path, **data)
    # os.replace(tmp_path, step_save_path)

    return t_idx


# Let's take this one for this example
# repo_id = "lerobot/aloha_mobile_cabinet"
repo_id = "sriramsk/plate_table_multiview_20251113_ss_hg"
repo_id = "sriramsk/fold_towel_MV_20251210_ss_hg"
repo_id = "sriramsk/fold_onesie_MV_20251210_ss_hg"
repo_id = "sriramsk/fold_onesie_MV_20260119_ss_hg"
# We can have a look and fetch its metadata to know more about it:
ds_meta = LeRobotDatasetMetadata(repo_id)

# Or simply load the entire dataset:
dataset = LeRobotDataset(repo_id, tolerance_s=0.0005)
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")

episode_index = 0
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

### NOTE: load camera intrinsics and extrinsics
### TODO: scale the intrinsics if we resize the depth images
all_intrinsics = []
all_extrinsics = []
cameras = {
  "cam_azure_kinect_front": {
    "intrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/intrinsics_000259921812.txt",
    # "extrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_front_v1_1020.txt"
    "extrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_front_20260121.txt"
  },
  "cam_azure_kinect_back": {
    "intrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/intrinsics_000003493812.txt",
    "extrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_back_v1_1020.txt"
  }
}
for cam_name, cam_cfg in cameras.items():
    # Load intrinsics
    K = _load_camera_intrinsics(cam_cfg['intrinsics'])
    orig_shape = [720, 1280]
    K_scaled = get_scaled_intrinsics(
        K, orig_shape, target_shape
    )
    all_intrinsics.append(K_scaled)
    print(f"Loaded intrinsics for cam_name: {cam_name}: {K}")

    # Load extrinsics
    T = _load_camera_extrinsics(cam_cfg['extrinsics'])
    all_extrinsics.append(T)
    print(f"Loaded extrinsics for cam_name: {cam_name}: {T}")

num_points = 4500
max_depth = 1.5
store_dino = True

depth_keys = [
    "observation.images.cam_azure_kinect_front.transformed_depth",
    "observation.images.cam_azure_kinect_back.transformed_depth",
]
color_keys = [
    "observation.images.cam_azure_kinect_front.color",
    "observation.images.cam_azure_kinect_back.color",
]

for traj_idx in range(dataset.num_episodes):
# for traj_idx in range(1):
    
    traj_eef_pose = []
    traj_gripper_pcd = []
    traj_scene_pcd = []
    traj_goal_gripper_pcd = []
    traj_fpsed_rgb_features = []
    traj_fpsed_rgb_values = []
    traj_wrist_images = []
    traj_rgb_images_front = []
    traj_rgb_images_back = []
    
    from_idx = dataset.episode_data_index["from"][traj_idx].item()
    to_idx = dataset.episode_data_index["to"][traj_idx].item()
    traj_len = to_idx - from_idx
    print(f"Trajectory {traj_idx} has length {traj_len} from {from_idx} to {to_idx}")

    NUM_WORKERS = 20 # try 2-8 depending on CPU/GPU + IO

    # -------------------- usage inside your traj loop --------------------
    traj_len = to_idx - from_idx
    results = [None] * traj_len

    beg = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = {
            ex.submit(
                process_one_frame,
                dataset,
                t_idx,
                traj_idx,
                from_idx,
                all_intrinsics,
                all_extrinsics,
                max_depth,
                num_points,
                store_dino,
                target_shape,
            ): t_idx
            for t_idx in range(from_idx, to_idx)
        }

        for fut in as_completed(futs):
            t_idx = futs[fut]
            out = fut.result()
            results[t_idx - from_idx] = out

    # Rebuild your original lists in-order
    traj_wrist_images = [r["wrist_image"] for r in results]
    traj_rgb_images_front = [r["rgb_image_front"] for r in results]
    traj_rgb_images_back = [r["rgb_image_back"] for r in results]
    cprint(f"Finished processing traj {traj_idx} in {time.time() - beg:.2f} seconds", "green")

    # data_dir = "/data/yufei/lerobot/data/plate_new_rot_rgb"
    # data_dir = "/data/yufei/lerobot/data/towel_1210_rgb"
    # data_dir = "/data/yufei/lerobot/data/onesie_1210_rgb_dino"
    combine_action_steps = 1
    data_dir = "/data/yufei/lerobot/data/onesie_0121_rgb_dino_{}".format(combine_action_steps)
    traj_dir = os.path.join(data_dir, f"traj_{traj_idx:04d}")
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)

    # ---- parallel save ----
    
    num_steps = len(traj_wrist_images) - 1
    saved_num_steps = len(os.listdir(traj_dir))
    assert saved_num_steps == num_steps, f"Expected {num_steps} steps but found {saved_num_steps - 1} in {traj_dir}"
    # import pdb; pdb.set_trace()
    
    SAVE_WORKERS = 20  # adjust based on your IO speed and CPU
    beg = time.time()
    
    
    with ThreadPoolExecutor(max_workers=SAVE_WORKERS) as ex:
        futs = [
            ex.submit(
                _save_step_npz,
                t_idx,
                traj_dir,
                traj_wrist_images,
                traj_rgb_images_front,
                traj_rgb_images_back,
            )
            for t_idx in range(num_steps)
        ]

        for fut in as_completed(futs):
            t_idx_done = fut.result()  # raises if any error
            # (optional) print progress occasionally
            # print("saved", t_idx_done)
            
    cprint(f"Finished saving traj {traj_idx} in {time.time() - beg:.2f} seconds", "blue")