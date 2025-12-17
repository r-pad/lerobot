
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


### TODO: load a dino-v2 model
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
    base_rgb_features = traj_rgb_features[0]
    base_rgb_values = traj_rgb_values[0]
    
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
            print("setting large finger change at time step ", i)
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
            filtered_rgb_features.append(base_rgb_features)
            filtered_rgb_values.append(base_rgb_values)
            
            base_pc = traj_pc[i+1]
            base_gripper_pcd = traj_gripper_pcd[i+1]
            base_pos_ori = traj_pos_ori[i+1]
            base_pos = target_pos
            base_ori_6d = target_ori_6d
            base_finger_angle = target_finger_angle
            base_goal_gripper_pcd = traj_goal_gripper_pcd[i+1]
            base_rgb_features = traj_rgb_features[i+1]
            base_rgb_values = traj_rgb_values[i+1]

    return np.asarray(traj_actions), filtered_pcs, filtered_pos_oris, filtered_gripper_pcds, filtered_goal_gripper_pcds, filtered_rgb_values, filtered_rgb_features

# Let's take this one for this example
# repo_id = "lerobot/aloha_mobile_cabinet"
repo_id = "sriramsk/plate_table_multiview_20251113_ss_hg"
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
    "extrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_front_v1_1020.txt"
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

for traj_idx in range(dataset.num_episodes):
# for traj_idx in range(1):
    
    traj_eef_pose = []
    traj_gripper_pcd = []
    traj_scene_pcd = []
    traj_goal_gripper_pcd = []
    traj_fpsed_rgb_features = []
    traj_fpsed_rgb_values = []
    
    from_idx = dataset.episode_data_index["from"][traj_idx].item()
    to_idx = dataset.episode_data_index["to"][traj_idx].item()
    traj_len = to_idx - from_idx
    print(f"Trajectory {traj_idx} has length {traj_len} from {from_idx} to {to_idx}")

    ### TODO: add DINO-V2 features for the RGB images and downsample them to the same fpsed pcd    
    for t_idx in range(from_idx, to_idx):
        
        print(f"Processing traj {traj_idx}, frame_idx {t_idx - from_idx} / {traj_len}")
        
        dataset_idx = t_idx
        data_point = dataset[dataset_idx]
        
        right_eef_pose = data_point['observation.right_eef_pose']#.cpu().numpy()  # (state_dim,)     
        eef_pos, eef_rot_6d, eef_gripper_width, eef_pos_robot_base, eef_rot_matrix_robot_base, eef_rot_6d_robot_base, eef_gripper_width_franka = get_gripper_4_points_from_sriram_data(right_eef_pose)
        
        traj_eef_pose.append(np.array([*eef_pos_robot_base, *eef_rot_6d_robot_base, *eef_gripper_width_franka]))
        
        ### NOTE: perform the conversion from the gripper pose to the gripper 4 points
        beg = time.time()
        eef_4_points = get_4_points_from_gripper_pos_orient(eef_pos_robot_base, eef_rot_matrix_robot_base, eef_gripper_width_franka)
        end = time.time()
        # print("time for getting 4 points from gripper pose: ", end - beg)
        traj_gripper_pcd.append(eef_4_points)
    
        ### NOTE: get the scene point cloud from the depth image
        beg = time.time()
        depth_keys = ["observation.images.cam_azure_kinect_front.transformed_depth", "observation.images.cam_azure_kinect_back.transformed_depth"]
        all_cam_depth_images = []
        for depth_key in depth_keys:
            depth = Image.fromarray(data_point[depth_key].numpy()[0])
            ### scale the depth image
            depth = np.asarray(depth_preprocess(depth))
            all_cam_depth_images.append(depth)
        
        color_keys = ["observation.images.cam_azure_kinect_front.color", "observation.images.cam_azure_kinect_back.color"]
        all_cam_color_images = []
        for color_key in color_keys:
            rgb = (data_point[color_key].permute(1, 2, 0).numpy() * 255).astype(
                np.uint8
            )
            rgb = Image.fromarray(rgb)
            rgb = np.asarray(rgb_preprocess(rgb))
            all_cam_color_images.append(rgb)
            
        ### NOTE: encode dino-v2 features for the rgb images here
        all_rgb_flattend = [rgb.reshape(-1, 3) for rgb in all_cam_color_images]
        all_rgb_flattend = np.concatenate(all_rgb_flattend, axis=0)  # (num_cams * H * W, 3)
        all_rgb_features_flattened = []
        for rgb in all_cam_color_images:
            rgb_features = compute_dino_v2_features(rgb, target_shape=target_shape)
            all_rgb_features_flattened.append(rgb_features)    
        all_rgb_features_flattened = np.concatenate(all_rgb_features_flattened, axis=0)  # (num_cams * H * W, feat_dim)
            
        # import pdb; pdb.set_trace()    
        # %matplotlib inline
        import matplotlib.pyplot as plt
        # plt.imshow(all_cam_depth_images[0])
        # plt.show()

        # %matplotlib widget
        from matplotlib import pyplot as plt
        all_pcd_in_world = []
        depth_masks = []
        for (depth, intrisincs, extrinsics) in zip(all_cam_depth_images, all_intrinsics, all_extrinsics):
            pcd_in_camera, depth_mask = get_scene_pcd_cam_frame(
                depth, intrisincs, None, max_depth
            )

            depth_masks.append(depth_mask.flatten())
            
            pcd_in_world = transform_to_world_frame(pcd_in_camera, extrinsics)
            x = pcd_in_world

            ### use open3d to visualize the point cloud
            # x = pcd_in_world[pcd_in_world[:, 1] < 0.6]
            # x = x[x[:, 1] > -0.4]
            
            x_in_robot_base = transform_from_table_center_to_robot_base(x)
            
            # fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
            # ### Plot in table coordinate
            # ax = axes[0]
            # ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=1, color='blue')
            # ax.scatter([0], [0], [0], s=50, color='red')
            # ax.scatter([eef_pos[0]], [eef_pos[1]], [eef_pos[2]], s=50, color='red')
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            
            # ### plot in robot base coordinate
            # ax = axes[1]
            # ax.scatter(x_in_robot_base[:, 0], x_in_robot_base[:, 1], x_in_robot_base[:, 2], s=1, color='blue')
            # ax.scatter([0], [0], [0], s=50, color='red')
            # ax.scatter([eef_pos_robot_base[0]], [eef_pos_robot_base[1]], [eef_pos_robot_base[2]], s=50, color='red')
            # ax.scatter(eef_4_points[:, 0], eef_4_points[:, 1], eef_4_points[:, 2], s=10, color='green')
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            
            # plt.show()
        
            # import pdb; pdb.set_trace()
            all_pcd_in_world.append(x_in_robot_base)
        
        all_pcd_in_world = np.concatenate(all_pcd_in_world, axis=0)  # (num_cams * num_points, 3)
        depth_masks = np.concatenate(depth_masks, axis=0)  # (num_cams * H * W,)
        all_rgb_features_flattened = all_rgb_features_flattened[depth_masks]
        all_rgb_flattend = all_rgb_flattend[depth_masks]


        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(all_pcd_in_world)
        # pcd1.paint_uniform_color([1, 0, 0])   # Red

        # geometries = [pcd1]

        # # --- Create spheres for pcd2 ---
        # sphere_radius = 0.01
        # sphere_resolution = 10

        # for p in eef_4_points:
        #     sphere = o3d.geometry.TriangleMesh.create_sphere(
        #         radius=sphere_radius, 
        #         resolution=sphere_resolution
        #     )
        #     sphere.paint_uniform_color([0, 0, 1])  # red spheres
        #     sphere.translate(p)                    # move sphere to point location
        #     geometries.append(sphere)

        # # Visualize
        # o3d.visualization.draw_geometries(geometries)

        
        ### perform fps here
        beg = time.time()

        filter_idx_1 = all_pcd_in_world[:, 1] < 0.6
        filter_idx_2 = all_pcd_in_world[:, 1] > -0.4
        filter_idx = np.logical_and(filter_idx_1, filter_idx_2)
        all_pcd_in_world = all_pcd_in_world[filter_idx]
        all_rgb_features_flattened = all_rgb_features_flattened[filter_idx]
        all_rgb_flattend = all_rgb_flattend[filter_idx]


        all_pcd_in_world, fps_index = fpsample_pcd(all_pcd_in_world, num_points)
        rgb_features_fpsed = all_rgb_features_flattened[fps_index]
        rgb_value_fpsed = all_rgb_flattend[fps_index]  
        end = time.time()
        # print("time for fps of the scene pcd: ", end - beg)
        
        # print("all_pcd_in_world shape:", all_pcd_in_world.shape)
        traj_scene_pcd.append(all_pcd_in_world)
        traj_fpsed_rgb_features.append(rgb_features_fpsed)
        traj_fpsed_rgb_values.append(rgb_value_fpsed)
        # import pdb; pdb.set_trace()
        # ax = plt.subplot(projection='3d')
        # ax.scatter(all_pcd_in_world[:, 0], all_pcd_in_world[:, 1], all_pcd_in_world[:, 2], s=1, color=rgb_value_fpsed.astype(np.float32)/255.0)
        # plt.show()
        

        ### use open3d to visualize the colored pcd
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(all_pcd_in_world)
        # pcd.colors = o3d.utility.Vector3dVector(rgb_value_fpsed.astype(np.float32)/255.0)
        # o3d.visualization.draw_geometries([pcd])


        ### NOTE: get the goal event idx, goal eef pose and convert that to the goal gripper pcd
        episode_index = data_point["episode_index"].item()
        assert episode_index == traj_idx
        assert data_point["frame_index"].item() == t_idx - from_idx
        # The next_event_idx is relative to the episode, so we calculate the absolute index
        goal_idx = (
            data_point["next_event_idx"] - data_point["frame_index"] + dataset_idx
        ).item()
        
        goal_idx_2 = data_point['next_event_idx'].item() + from_idx
        assert goal_idx == goal_idx_2

        # print("relative first goal idx is: ", data_point["next_event_idx"] )
        # print("first goal idx is: ", goal_idx)

        # continue
        
        goal_data_point = dataset[goal_idx]
        # rgb_image = Image.fromarray(goal_data_point['observation.images.cam_azure_kinect_front.color'].numpy()[0])
        # rgb_image = np.asarray(rgb_image)
        # %matplotlib inline
        # import matplotlib.pyplot as plt
        # plt.imshow(rgb_image)
        # plt.show()
        goal_right_eef_pose = goal_data_point['observation.right_eef_pose']  # (state_dim,)

        # depth_key = "observation.images.cam_azure_kinect_front.transformed_depth"
        # goal_depth = Image.fromarray(goal_data_point[depth_key].numpy()[0])
        # goal_depth = np.asarray(goal_depth)
        # goal_pcd_in_camera = get_scene_pcd_cam_frame(
        #         goal_depth, all_intrinsics[0], num_points, max_depth
        # )
        # goal_pcd_in_world = transform_to_world_frame(goal_pcd_in_camera, all_extrinsics[0])
        # ### use open3d to visualize the point cloud
        # goal_x = goal_pcd_in_world[goal_pcd_in_world[:, 1] < 0.6]
        # goal_x = goal_x[goal_x[:, 1] > -0.4]
        # goal_x_in_robot_base = transform_from_table_center_to_robot_base(goal_x)
        
        beg = time.time()
        goal_eef_pos, goal_eef_rot_6d, goal_eef_gripper_width, goal_eef_pos_robot_base, goal_eef_rot_matrix_robot_base, goal_eef_rot_6d_robot_base, goal_eef_gripper_width_franka =\
            get_gripper_4_points_from_sriram_data(goal_right_eef_pose)
        goal_4_points = get_4_points_from_gripper_pos_orient(goal_eef_pos_robot_base, goal_eef_rot_matrix_robot_base, goal_eef_gripper_width_franka)
        traj_goal_gripper_pcd.append(goal_4_points)
        end = time.time()
        # print("time for getting goal gripper pcd: ", end - beg)
        
        # %matplotlib widget
        # from matplotlib import pyplot as plt
        # # plt.close("all")
        # ax = plt.subplot(projection='3d')
        # ax.scatter(all_pcd_in_world[:, 0], all_pcd_in_world[:, 1], all_pcd_in_world[:, 2], s=1, color='blue')
        # ax.scatter(goal_x_in_robot_base[:, 0], goal_x_in_robot_base[:, 1], goal_x_in_robot_base[:, 2], s=1, color='blue')
        # ax.scatter([eef_pos_robot_base[0]], [eef_pos_robot_base[1]], [eef_pos_robot_base[2]], s=50, color='red')
        # ax.scatter(eef_4_points[:, 0], eef_4_points[:, 1], eef_4_points[:, 2], s=10, color='green')
        # ax.scatter([goal_eef_pos_robot_base[0]], [goal_eef_pos_robot_base[1]], [goal_eef_pos_robot_base[2]], s=50, color='red')
        # ax.scatter(goal_4_points[:, 0], goal_4_points[:, 1], goal_4_points[:, 2], s=50, color='black')
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # plt.show()

        # import pdb; pdb.set_trace()
        
    ### for action: check the magnitude of our dataset and make sure we have similar action magnitudes
    ### maybe merge actions
    # traj_eef_pose = np.stack(traj_eef_pose, axis=0)  # (traj_len, 10)
    # all_delta_translations = traj_eef_pose[1:, :3] - traj_eef_pose[:-1, :3]  # (traj_len - 1, 3)
    # all_delta_translations_magnitudes = np.linalg.norm(all_delta_translations, axis=1)  # (traj_len - 1,)
    # print("mean translation delta:", np.mean(all_delta_translations_magnitudes))
    # print("max translation delta:", np.max(all_delta_translations_magnitudes))
    # print("min translation delta:", np.min(all_delta_translations_magnitudes))
            
    '''
    for t_idx in range(traj_len):
        step_save_dir = os.path.join(save_dir, str(t_idx) + ".pkl")
            
        pickle_data = {}
        pickle_data['state'] = state_arrays[t_idx][None, :]
        pickle_data['point_cloud'] = point_cloud_arrays[t_idx][None, :]
        pickle_data['action'] = action_arrays[t_idx][None, :]
        pickle_data['gripper_pcd'] = gripper_pcd_arrays[t_idx][None, :]
        if goal_gripper_pcd is not None:
            pickle_data['goal_gripper_pcd'] = goal_gripper_pcd[t_idx][None, :]
        if displacement_gripper_to_object is not None:
            pickle_data['displacement_gripper_to_object'] = displacement_gripper_to_object[t_idx][None, :]
        if dp3_pc_list is not None:
            pickle_data['dp3_point_cloud'] = dp3_pc_list[t_idx][None, :]
        
        with open(step_save_dir, 'wb') as f:
            pickle.dump(pickle_data, f)
    '''

    ### TODO: extract the actions
    action_arrays, traj_scene_pcd, traj_eef_pose, traj_gripper_pcd, traj_goal_gripper_pcd, traj_fpsed_rgb_values, traj_fpsed_rgb_features = \
        extract_actions(traj_eef_pose, traj_scene_pcd, traj_gripper_pcd, traj_goal_gripper_pcd, traj_fpsed_rgb_values, traj_fpsed_rgb_features, combine_action_steps=2)

    data_dir = "/data/yufei/lerobot/data/plate_new_rot_rgb"
    traj_dir = os.path.join(data_dir, f"traj_{traj_idx:04d}")
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)

    for t_idx in range(len(traj_eef_pose)):
        # step_save_dir = os.path.join(traj_dir, str(t_idx) + ".pkl")
        step_save_dir = os.path.join(traj_dir, str(t_idx) + ".npz")
        pickle_data = {}
        pickle_data['state'] = traj_eef_pose[t_idx][None, :]
        pickle_data['point_cloud'] = traj_scene_pcd[t_idx][None, :]
        pickle_data['action'] = action_arrays[t_idx][None, :]
        pickle_data['gripper_pcd'] = traj_gripper_pcd[t_idx][None, :]
        pickle_data['goal_gripper_pcd'] = traj_goal_gripper_pcd[t_idx][None, :]
        pickle_data['rgb_values'] = traj_fpsed_rgb_values[t_idx][None, :].astype(np.float32) / 255.0
        pickle_data['rgb_features'] = traj_fpsed_rgb_features[t_idx][None, :]
        
        # with open(step_save_dir, 'wb') as f:
        #     pickle.dump(pickle_data, f)
        
        np.savez_compressed(step_save_dir, **pickle_data)