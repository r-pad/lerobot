from train_ddp import TrainDP3Workspace
import hydra
from omegaconf import OmegaConf
from copy import deepcopy
import torch
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix   
from pytorch3d.ops import sample_farthest_points
from termcolor import cprint
import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

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

default_intrinsics = []
default_extrinsics = []
scaled_intrinsics = []
cameras = {
  "cam_azure_kinect_front": {
    "intrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/intrinsics_000259921812.txt",
    # "extrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_front_v1_1020.txt"
    "extrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_front_1208.txt"
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
    default_intrinsics.append(K)
    scaled_intrinsics.append(K_scaled)

    # Load extrinsics
    T = _load_camera_extrinsics(cam_cfg['extrinsics'])
    default_extrinsics.append(T)
    # print(f"Loaded extrinsics for cam_name: {cam_name}: {T}")


def low_level_policy_infer(obj_pcd, agent_pos, goal_gripper_pcd, gripper_pcd, policy, cat_idx=13):
    input_dict = {
        "point_cloud": obj_pcd,
        "agent_pos": agent_pos,
        'gripper_pcd': gripper_pcd,
        'goal_gripper_pcd': goal_gripper_pcd,
    }

    batched_action = policy.predict_action(input_dict, torch.tensor([cat_idx]).to(policy.device))

    return batched_action['action'] # B, T, 10

def load_low_level_policy(exp_dir, checkpoint_name):
    with hydra.initialize(config_path='../../3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(exp_dir)),
        )
        cfg = recomposed_config
        
    workspace = TrainDP3Workspace(cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir)

    policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        policy = deepcopy(workspace.ema_model)
    policy.eval()
    policy.reset()
    policy = policy.to('cuda')
    
    return policy


def load_multitask_high_level_model(path):
    from omegaconf import OmegaConf
    import json
    ckpt_path = os.path.dirname(path)
    config_path = os.path.join(ckpt_path, "config.json")
    cfg = json.load(open(config_path, "r"))
    cfg = OmegaConf.create(cfg)
    args = cfg
    
    device = torch.device("cuda")
    general_args = args.general
    input_channel = 5 if general_args.add_one_hot_encoding else 3
    if general_args.get("use_rgb", False):
        input_channel += 3
    if general_args.get("use_dino", False):
        input_channel += 1024

    output_dim = 13 
    from test_PointNet2.model_invariant import PointNet2_super_multitask
    
    if "category_embedding_type" not in general_args:
        general_args.category_embedding_type = None
    if general_args.category_embedding_type == "one_hot":
        embedding_dim = args.num_categories
    elif general_args.category_embedding_type == "siglip":
        embedding_dim = 768
    else:
        embedding_dim = None
    
    model = PointNet2_super_multitask(num_classes=output_dim, keep_gripper_in_fps=general_args.keep_gripper_in_fps, input_channel=input_channel,
                                      first_sa_point=general_args.get("first_sa_point", 2048),
                                      fp_to_full=general_args.get("fp_to_full", False),
                                      replace_bn_w_gn=general_args.get("replace_bn_with_gn", False),
                                      replace_bn_w_in=general_args.get("replace_bn_with_in", False),
                                      embedding_dim=embedding_dim,
                                      film_in_sa_and_fp=general_args.get("film_in_sa_and_fp", False),
                                      embedding_as_input=general_args.get("embedding_as_input", False),
                                      replace_bn_w_ln=general_args.get("replace_bn_with_ln", False),
                                      ).to(device)
    
    model.load_state_dict(torch.load(path, map_location=device)['model'])
    print("Successfully load model from: ", path)
    model.eval()
        
    return model, args

def infer_multitask_high_level_model(inputs, goal_prediction_model, cat_embedding=None, high_level_args=None, extra=None):
    if high_level_args.get("use_rgb", False):
        rgb = extra['rgb']
        gripper_rgb = extra['rgb_gripper']
        extra_rgb = torch.cat([rgb, gripper_rgb], dim=1)
        inputs = torch.cat([inputs, extra_rgb], dim=2)  # B, N+4, 6
    if high_level_args.get("use_dino", False):
        dino_features = extra['dino_features']
        gripper_dino_features = extra['dino_features_gripper']
        extra_dino_features = torch.cat([dino_features, gripper_dino_features], dim=1)
        inputs = torch.cat([inputs, extra_dino_features], dim=2)  # B, N+4, 6 + 1024


    if high_level_args is not None:
        if high_level_args.add_one_hot_encoding:
            print("adding one hot encoding to the input")
            N_scene_points = inputs.shape[1] - 4
            pointcloud_one_hot = torch.zeros(inputs.shape[0], inputs.shape[1], 2).float().to(inputs.device)
            pointcloud_one_hot[:, :N_scene_points, 0] = 1
            pointcloud_one_hot[:, N_scene_points:, 1] = 1
            inputs = torch.cat([inputs, pointcloud_one_hot], dim=2) # B, N+4, 5
    
    inputs = inputs.to('cuda')
    inputs_ = inputs.permute(0, 2, 1)
    with torch.no_grad():
        pred_dict = goal_prediction_model(inputs_, cat_embedding, build_grasp=False, articubot_format=True) 
    outputs = pred_dict['pred_offsets']
    pred_points = pred_dict['pred_points'] 
    weights = pred_dict['pred_scores'].squeeze(-1)
    inputs = pred_points
    B, N, _, _ = outputs.shape
    outputs = outputs.view(B, N, -1)
    
    outputs = outputs.view(B, N, 4, 3)
    
    ### sample an displacement according to the weight
    probabilities = weights  # Must sum to 1
    probabilities = torch.nn.functional.softmax(weights, dim=1)

    # Sample one index based on the probabilities
    sampled_index = torch.argmax(probabilities.squeeze(0))

    displacement_mean = outputs[:, sampled_index, :, :] # B, 4, 3
    input_point_pos = inputs[:, sampled_index, :] # B, 3
    prediction = input_point_pos.unsqueeze(1) + displacement_mean # B, 4, 3
        
    return prediction



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

def transform_from_robot_base_to_table_center(xyz_in_robot_base):
    T_robot_to_table_world = np.array([
        [-1, 0, 0, 0.449],
        [0, -1, 0, -0.019,],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    N = xyz_in_robot_base.shape[0]
    homo = np.ones((N, 1))
    xyz_in_robot_base_homo = np.concatenate([xyz_in_robot_base, homo], axis=1) # N x 4
    xyz_in_table_center_homo = (T_robot_to_table_world @ xyz_in_robot_base_homo.T).T
    xyz_in_table_center = xyz_in_table_center_homo[:, :3]
    return xyz_in_table_center

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

def eef_matrix_robot_base_to_aloha_eef_matrix(eef_rot_matrix_robot_base):
    # eef_rot_matrix_franka_eef_coordinate = eef_rot_matrix @ R_vf_y @ R_vf_z
    # eef_rot_matrix_robot_base = R_world_to_robot @ eef_rot_matrix_franka_eef_coordinate
    
    ### convert from robot base frame to table center frame
    eef_table_center_franka_matrix = R_world_to_robot.T @ eef_rot_matrix_robot_base

    ### convert from franka eef coordinate to aloha eef coordinate
    eef_matrix_aloha = eef_table_center_franka_matrix @ R_vf_z.T @ R_vf_y.T
    
    return eef_matrix_aloha

def get_gripper_4_points_from_sriram_data(eef_pose_from_sriram):
    right_eef_pose = eef_pose_from_sriram
    eef_rot_6d = right_eef_pose[:6]
    eef_rot_matrix = rotation_6d_to_matrix(eef_rot_6d[None, :]).squeeze().cpu().numpy()  # (3, 3)
    
    ### convert from aloha eef coordinate to franka eef coordinate
    eef_rot_matrix_franka_eef_coordinate = eef_rot_matrix @ R_vf_y @ R_vf_z
    ### convert from table center to robot base frame
    eef_rot_matrix_robot_base = R_world_to_robot @ eef_rot_matrix_franka_eef_coordinate
    eef_rot_6d_robot_base = matrix_to_rotation_6d(torch.from_numpy(eef_rot_matrix_robot_base)[None, :]).squeeze().cpu().numpy()
    
    eef_pos = right_eef_pose[6:9].cpu().numpy()
    eef_pos_robot_base = transform_from_table_center_to_robot_base(eef_pos[None, :]).squeeze()
    
    ### NOTE: perform the conversion of the aloha eef gripper width to panda gripper width
    eef_gripper_width = right_eef_pose[9:10].cpu().numpy()
    eef_gripper_width_franka = eef_gripper_width / 80 * 0.04  # convert to franka gripper width
    eef_gripper_width_franka = np.clip(eef_gripper_width_franka, 0, 0.04)

    return eef_pos, eef_rot_6d, eef_gripper_width, eef_pos_robot_base, eef_rot_matrix_robot_base, eef_rot_6d_robot_base, eef_gripper_width_franka
    
def compute_pcd(all_cam_depth_images, all_intrinsics=None, all_extrinsics=None, max_depth=1.5, num_points=4500, all_cam_rgb_images=None, use_dino=False):
    if all_intrinsics is None or all_extrinsics is None:
        all_intrinsics = default_intrinsics
        all_extrinsics = default_extrinsics
    if all_cam_rgb_images is not None:
        all_intrinsics = scaled_intrinsics

    all_pcd_in_world = []
    all_pcd_in_table_center = []
    depth_masks = []


    for (depth, intrisincs, extrinsics) in zip(all_cam_depth_images, all_intrinsics, all_extrinsics):
        depth = depth / 1000.0
        pcd_in_camera, depth_mask = get_scene_pcd_cam_frame(
            depth, intrisincs, None, max_depth
        )

        depth_masks.append(depth_mask.flatten())
        

        # import pdb; pdb.set_trace()
        pcd_in_world = transform_to_world_frame(pcd_in_camera, extrinsics)
        x = pcd_in_world
        
        ### use open3d to visualize the point cloud
        # x = pcd_in_world[pcd_in_world[:, 1] < 0.6]
        # x = x[x[:, 1] > -0.4]
        
        all_pcd_in_table_center.append(x)
        x_in_robot_base = transform_from_table_center_to_robot_base(x)
        
    
        # import pdb; pdb.set_trace()
        all_pcd_in_world.append(x_in_robot_base)

    depth_masks = np.concatenate(depth_masks, axis=0)  # (num_cams * H * W,)
    # import pdb; pdb.set_trace()
    if all_cam_rgb_images is not None:
        all_rgb_flattend = [rgb.reshape(-1, 3) for rgb in all_cam_rgb_images]
        all_rgb_flattend = np.concatenate(all_rgb_flattend, axis=0)  # (num_cams * H * W, 3)
        all_rgb_flattend = all_rgb_flattend[depth_masks]


    all_dino_features = None
    # import pdb; pdb.set_trace()
    if use_dino and all_cam_rgb_images is not None:
        all_dino_features = [compute_dino_v2_features(rgb) for rgb in all_cam_rgb_images]
        all_dino_features = np.concatenate(all_dino_features, axis=0)  # (num_cams * H * W, feat_dim)
        all_dino_features = all_dino_features[depth_masks]
    
    all_pcd_in_world = np.concatenate(all_pcd_in_world, axis=0)  # (num_cams * num_points, 3)
    all_pcd_in_table_center = np.concatenate(all_pcd_in_table_center, axis=0)

    filter_idx_1 = all_pcd_in_world[:, 1] < 0.6
    filter_idx_2 = all_pcd_in_world[:, 1] > -0.4
    filter_idx = np.logical_and(filter_idx_1, filter_idx_2)
    all_pcd_in_world = all_pcd_in_world[filter_idx]
    all_pcd_in_table_center = all_pcd_in_table_center[filter_idx]
    if all_cam_rgb_images is not None:
        all_rgb_flattend = all_rgb_flattend[filter_idx]
    if use_dino and all_cam_rgb_images is not None:
        all_dino_features = all_dino_features[filter_idx]
    
    all_pcd_in_world, fps_idx = fpsample_pcd(all_pcd_in_world, num_points)
    all_pcd_in_table_center = all_pcd_in_table_center[fps_idx]
    if all_cam_rgb_images is not None:
        all_rgb_flattend = all_rgb_flattend[fps_idx]
        all_rgb_flattend = all_rgb_flattend.astype(np.float32) / 255.0
        if use_dino:
            all_dino_features = all_dino_features[fps_idx]
    else:
        all_rgb_flattend = None
    
    return all_pcd_in_world, all_pcd_in_table_center, all_rgb_flattend, all_dino_features


def get_aloha_future_eef_poses_from_delta_actions(low_level_action, 
                                                  eef_pos_robot_base, 
                                                  eef_rot_matrix_robot_base, 
                                                  eef_gripper_width_franka):
    low_level_action = low_level_action.squeeze(0).cpu().numpy()  # 4 x 10
    cur_eef_pos = copy.deepcopy(eef_pos_robot_base)
    cur_eef_matrix = copy.deepcopy(eef_rot_matrix_robot_base)
    cur_gripper_width = copy.deepcopy(eef_gripper_width_franka)
    eef_pos = []
    eef_orient_matrix = []
    gripper_widths = []
    for act in low_level_action:
        delta_pos = act[0:3]
        delta_rot_6d = act[3:9]
        delta_gripper_width = act[9]

        new_eef_pos = cur_eef_pos + delta_pos
        new_eef_rot_matrix = cur_eef_matrix @ rotation_transfer_6D_to_matrix(delta_rot_6d)
        # new_eef_rot_matrix = cur_eef_matrix # @ rotation_transfer_6D_to_matrix(delta_rot_6d)
        new_gripper_width = cur_gripper_width + delta_gripper_width
        new_gripper_width= np.clip(new_gripper_width, 0.0, 0.04)

        cur_eef_pos = new_eef_pos
        cur_eef_matrix = new_eef_rot_matrix
        cur_gripper_width = new_gripper_width
        eef_pos.append(new_eef_pos)
        eef_orient_matrix.append(new_eef_rot_matrix)
        gripper_widths.append(new_gripper_width)
    
    eef_pos = np.array(eef_pos).reshape(-1, 3)
    # eef_orient_matrix = np.array(eef_orient_matrix)
    gripper_widths = np.array(gripper_widths)
    
    aloha_world_eef_pos = transform_from_robot_base_to_table_center(eef_pos)
    aloha_world_eef_orient_matrix = [eef_matrix_robot_base_to_aloha_eef_matrix(mat) for mat in eef_orient_matrix]
    aloha_world_eef_orient_6d = [rotation_transfer_matrix_to_6D(mat) for mat in aloha_world_eef_orient_matrix]
    aloha_gripper_widths = [x / 0.04 * 80 for x in gripper_widths]  # convert back to aloha gripper width
    cprint("gripper width: {}".format(aloha_gripper_widths), "yellow")

    
    return aloha_world_eef_pos, aloha_world_eef_orient_6d, aloha_gripper_widths, eef_pos
                    