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

import numpy as np
from scipy.spatial import cKDTree
import time
import cv2
from pyk4a import calibration

def transform_world_to_cam_frame(points_world: np.ndarray, T_world_from_cam: np.ndarray) -> np.ndarray:
    """
    Inverse of transform_to_world_frame: world -> camera.

    Args:
        points_world: (N, 3) points in world frame
        T_world_from_cam: (4, 4) world-from-cam transform

    Returns:
        points_cam: (N, 3) points in camera frame
    """
    T_cam_from_world = np.linalg.inv(T_world_from_cam)
    N = points_world.shape[0]
    pts_h = np.concatenate([points_world, np.ones((N, 1), dtype=points_world.dtype)], axis=1)  # (N,4)
    pts_cam_h = (T_cam_from_world @ pts_h.T).T
    return pts_cam_h[:, :3]

def project_world_pcd_to_mask(robot_pcd, extrinsics, camera, H, W):
    robot_pcd = np.concatenate([robot_pcd, np.ones((robot_pcd.shape[0], 1), dtype=robot_pcd.dtype)], axis=1)  # N x 4
    cam_aligned_robot_pcd = (np.linalg.inv(extrinsics) @ robot_pcd.T).T # to camera frame
    cam_aligned_robot_pcd = cam_aligned_robot_pcd[:,:3] / cam_aligned_robot_pcd[:,[3]]
    depth2cam_transform = camera.calibration.get_camera_matrix(calibration.CalibrationType.DEPTH)
    pixels = depth2cam_transform @ cam_aligned_robot_pcd.T
    pixels = pixels / pixels[2]
    pixels = np.rint(pixels).astype(np.int32)
    valid = (pixels[0] >= 0) & (pixels[0] < W) & (pixels[1] >= 0) & (pixels[1] < H)
    pixels = pixels[:, valid]
    x,y = pixels[0], pixels[1]
    mask = np.zeros((H,W), dtype=bool) # TODO: make this work for different resolutions
    mask[y,x] = True
    # kernel = np.ones((11, 11), np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    # kernel = np.ones((13, 13), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=5)
    mask = mask > 0

    return mask

def project_world_pcd_to_depth(
    points_world: np.ndarray,
    T_world_from_cam: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    max_depth: float = None,
    min_depth: float = 1e-6,
    invalid_value: float = 0.0,
    rounding: str = "round",   # "round" | "floor"
) -> np.ndarray:
    """
    Project a world-frame point cloud into a depth image using a z-buffer.

    Args:
        points_world: (N, 3) points in world frame
        T_world_from_cam: (4, 4) transform (world from cam)
        K: (3, 3) intrinsics
        height, width: output depth image size
        max_depth: if not None, discard points with z > max_depth
        min_depth: discard points with z <= min_depth (behind / too close)
        invalid_value: value to fill where no depth lands (commonly 0)
        rounding: how to convert projected float pixels to integer indices

    Returns:
        depth: (H, W) depth image in the *camera* z metric (same units as points)
    """
    # 1) world -> camera
    pts_cam = transform_world_to_cam_frame(points_world, T_world_from_cam)  # (N,3)
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    # 2) filter valid depths
    valid = z > min_depth
    if max_depth is not None:
        valid &= z < max_depth
    if not np.any(valid):
        return np.full((height, width), invalid_value, dtype=np.float32)

    x, y, z = x[valid], y[valid], z[valid]

    # 3) camera 3D -> pixel
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    if rounding == "floor":
        ui = np.floor(u).astype(np.int32)
        vi = np.floor(v).astype(np.int32)
    else:
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)

    # 4) keep only pixels inside image
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not np.any(inside):
        return np.full((height, width), invalid_value, dtype=np.float32)

    ui, vi, z = ui[inside], vi[inside], z[inside].astype(np.float32)

    # 5) z-buffer: for each pixel keep nearest (min z)
    depth = np.full((height, width), np.inf, dtype=np.float32)

    # np.minimum.at does an in-place grouped min reduction
    np.minimum.at(depth, (vi, ui), z)

    # 6) fill invalid pixels
    depth[~np.isfinite(depth)] = invalid_value
    return depth

def filter_pcd_a_by_b_radius(pcd_a, pcd_b, thresh=0.005):
    import pdb; pdb.set_trace()
    beg = time.time()
    tree = cKDTree(pcd_b)
    cprint("build kdtree time: {}".format(time.time() - beg), "yellow")
    import pdb; pdb.set_trace()
    # returns list of neighbor indices within radius for each point
    beg = time.time()
    neighbors = tree.query_ball_point(pcd_a, r=thresh, workers=-1)
    cprint("query ball point time: {}".format(time.time() - beg), "yellow")
    keep = np.fromiter((len(n) == 0 for n in neighbors), dtype=bool, count=len(neighbors))
    return keep

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

def compute_dino_v2_features(rgb, target_shape=224, device="cuda"):
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


def _is_numpy_image(x):
    return isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[2] == 3


def _to_pil_image(img):
    if isinstance(img, Image.Image):
        return img
    elif _is_numpy_image(img):
        if img.dtype != np.uint8:
            # convert safely to uint8 if needed
            img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)
    else:
        raise TypeError(
            f"Expected PIL.Image.Image or RGB numpy array [H, W, 3], got {type(img)}"
        )


def _normalize_input(images):
    """
    Normalize input to a list of PIL images and remember whether the input was single.
    """
    if isinstance(images, (Image.Image, np.ndarray)):
        return [_to_pil_image(images)], True

    if isinstance(images, (list, tuple)):
        if len(images) == 0:
            raise ValueError("Input list is empty.")
        pil_images = [_to_pil_image(img) for img in images]
        return pil_images, False

    raise TypeError(
        "Input must be a PIL image, numpy RGB image, or a list/tuple of them."
    )


def extract_dinov2_features(
    images,
    device="cuda",
    target_shape=224,
    flatten=False,
    return_numpy=True,
):
    """
    Unified API for extracting DINOv2 dense image features.

    Args:
        images:
            - single PIL.Image.Image
            - single numpy RGB image [H, W, 3]
            - list of PIL images
            - list of numpy RGB images
        dinov2:
            Preloaded DINOv2 model. If None, loads one.
        device:
            e.g. "cuda" or "cpu"
        target_shape:
            Resize + center crop target resolution.
        flatten:
            If False:
                returns dense feature maps
                single input  -> [H, W, C]
                batch input   -> [B, H, W, C]
            If True:
                returns flattened spatial features
                single input  -> [H*W, C]
                batch input   -> [B, H*W, C]
        return_numpy:
            If True, returns numpy arrays.
            If False, returns torch tensors.

    Returns:
        Features in either single-image or batched format depending on input.
    """
    pil_images, is_single = _normalize_input(images)

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                target_shape, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(target_shape),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    # [B, 3, H, W]
    inputs = torch.stack([preprocess(img) for img in pil_images], dim=0).to(device)

    with torch.no_grad():
        outputs = dinov2.forward_features(inputs)

    # [B, num_patches, C]
    patch_features = outputs["x_norm_patchtokens"]
    B, num_patches, feat_dim = patch_features.shape

    h = w = int(num_patches ** 0.5)
    if h * w != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a perfect square.")

    # [B, h, w, C]
    patch_features = patch_features.reshape(B, h, w, feat_dim)

    # [B, C, h, w]
    patch_features = patch_features.permute(0, 3, 1, 2)

    # [B, C, target_shape, target_shape]
    resized_features = F.interpolate(
        patch_features,
        size=(target_shape, target_shape),
        mode="bilinear",
        align_corners=False,
    )

    # [B, H, W, C]
    resized_features = resized_features.permute(0, 2, 3, 1)

    if flatten:
        # [B, H*W, C]
        resized_features = resized_features.reshape(B, target_shape * target_shape, feat_dim)

    if return_numpy:
        resized_features = resized_features.cpu().numpy()

    if is_single:
        return resized_features[0]

    return resized_features


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
    # "extrinsics": "/data/yufei/lerobot/lerobot/scripts/aloha_calibration/T_world_from_camera_front_1208.txt"
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
    default_intrinsics.append(K)
    scaled_intrinsics.append(K_scaled)

    # Load extrinsics
    T = _load_camera_extrinsics(cam_cfg['extrinsics'])
    default_extrinsics.append(T)
    # print(f"Loaded extrinsics for cam_name: {cam_name}: {T}")

# new_front_calibration = np.load("/data/yufei/lerobot/data/calibration/calibration_results/camcam_azure_kinect_front_calibration.npz")['T']
# new_back_calibration = np.load("/data/yufei/lerobot/data/calibration/calibration_results/camcam_azure_kinect_back_calibration.npz")['T']
# default_extrinsics[0] = new_front_calibration
# default_extrinsics[1] = new_back_calibration
all_cam_names = ["cam_azure_kinect_front", "cam_azure_kinect_back"]

camera_alignments = {}
alignment_path = "/data/yufei/lerobot/data/calibration/camera_alignments.npz"
data = np.load(alignment_path)
# The cam ids are stored as strings in the npz file, convert to int
for cam_name, transform in data.items():
    camera_alignments[cam_name] = transform


def low_level_policy_infer(obj_pcd, agent_pos, goal_gripper_pcd, gripper_pcd, policy, cat_idx=13):
    input_dict = {
        "point_cloud": obj_pcd,
        "agent_pos": agent_pos,
        'gripper_pcd': gripper_pcd,
        'goal_gripper_pcd': goal_gripper_pcd,
    }

    batched_action = policy.predict_action(input_dict, torch.tensor([cat_idx]).to(policy.device))
    # import pdb; pdb.set_trace()

    # return batched_action['action'] # B, T, 10
    return batched_action['action_pred'] # B, T, 10

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

def apply_transform(points, transform):
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # (N, 4)
    transformed_points_homo = (transform @ points_homo.T).T
    transformed_points = transformed_points_homo[:, :3]
    return transformed_points


def compute_pcd(all_cam_depth_images=None, all_pcds=None, all_intrinsics=None, all_extrinsics=None, max_depth=1.5, num_points=4500, all_cam_rgb_images=None, use_dino=False, 
                robot_pc=None, robot=None, debug_depth=None):
    if all_intrinsics is None or all_extrinsics is None:
        all_intrinsics = default_intrinsics
        all_extrinsics = default_extrinsics
    if all_cam_rgb_images is not None:
        all_intrinsics = scaled_intrinsics

    all_pcd_in_world = []
    all_pcd_in_table_center = []

    if all_cam_depth_images is not None:
        depth_masks = []
        for (depth, intrinsics, extrinsics) in zip(all_cam_depth_images, all_intrinsics, all_extrinsics):
            depth = depth / 1000.0


            if robot_pc is not None:
                right_robot_pc = robot_pc[robot_pc[:, 2] > 0.01]
                right_robot_pc = right_robot_pc[right_robot_pc[:, 0] > -0.2]

                robot_depth = project_world_pcd_to_depth(
                    right_robot_pc, extrinsics, intrinsics, depth.shape[0], depth.shape[1], max_depth=max_depth, invalid_value=0.0
                )

                robot_mask = robot_depth > 0.0
                ### dilate the mask a bit
                kernel = np.ones((11, 11), np.uint8)
                # kernel = np.ones((13, 13), np.uint8)
                robot_mask = cv2.dilate(robot_mask.astype(np.uint8), kernel, iterations=5)

                from matplotlib import pyplot as plt
                fig, ax = plt.subplots(1, 3)
                ori_depth = depth.copy()
                ax[0].imshow(ori_depth)
                ax[1].imshow(robot_mask)
                depth[robot_mask > 0] = 0
                ax[2].imshow(depth)
                plt.show()

            pcd_in_camera, depth_mask = get_scene_pcd_cam_frame(
                depth, intrinsics, None, max_depth
            )

            depth_masks.append(depth_mask.flatten())
            

            # import pdb; pdb.set_trace()
            pcd_in_world = transform_to_world_frame(pcd_in_camera, extrinsics)
            x = pcd_in_world
            
            ### use open3d to visualize the point cloud
            # x = pcd_in_world[pcd_in_world[:, 1] < 0.6]
            # x = x[x[:, 1] > -0.4]
            
            all_pcd_in_table_center.append(x) ### aloha table center frame
            x_in_robot_base = transform_from_table_center_to_robot_base(x) ### aloha right arm robot base frame
            
        
            # import pdb; pdb.set_trace()
            all_pcd_in_world.append(x_in_robot_base)
    elif all_pcds is not None:
        depth_masks = []
        for pcd, extrinsics, cam_name, d_depth in zip(all_pcds, all_extrinsics, all_cam_names, debug_depth):
            pcd_in_world = transform_to_world_frame(pcd, extrinsics)
            alignment = camera_alignments[cam_name]
            pcd_in_world = apply_transform(pcd_in_world, alignment)
            
            if robot_pc is not None:
                right_robot_pc = robot_pc[robot_pc[:, 2] > 0.01]
                right_robot_pc = right_robot_pc[right_robot_pc[:, 0] > -0.2]

                H, W = d_depth.shape
                print("H, W: ", H, W)
                robot_mask = project_world_pcd_to_mask(
                    right_robot_pc, extrinsics, robot.cameras[cam_name].camera, H, W
                )
                ### dilate the mask a bit
                kernel = np.ones((5, 5), np.uint8)
                # kernel = np.ones((13, 13), np.uint8)
                robot_mask = cv2.dilate(robot_mask.astype(np.uint8), kernel, iterations=5)
                robot_mask = robot_mask.flatten()

                # from matplotlib import pyplot as plt
                # fig, ax = plt.subplots(1, 2)
                # ori_depth = d_depth.copy()
                # ax[0].imshow(ori_depth)
                # ax[1].imshow(robot_mask.reshape(H, W))
                # plt.show()

                pcd_in_world = pcd_in_world[robot_mask == 0]
                depth_masks.append(robot_mask == 0)

            all_pcd_in_table_center.append(pcd_in_world) ### aloha table center frame
            x_in_robot_base = transform_from_table_center_to_robot_base(pcd_in_world) ### aloha
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
        beg = time.time()
        # all_dino_features = [compute_dino_v2_features(rgb, device="cuda") for rgb in all_cam_rgb_images]
        # all_dino_features = np.concatenate(all_dino_features, axis=0)  # (num_cams * H * W, feat_dim)
        # end = time.time()
        # cprint(f"compute dino features non-batched time: {end - beg} seconds", "blue")

        beg = time.time()
        all_dino_features_2 = extract_dinov2_features(all_cam_rgb_images, device="cuda", flatten=True, return_numpy=True)
        feature_dim = all_dino_features_2.shape[-1]
        all_dino_features_2 = all_dino_features_2.reshape(-1, feature_dim)
        all_dino_features = all_dino_features_2
        end = time.time()
        # diff = np.abs(all_dino_features - all_dino_features_2).mean()
        # cprint(f"dino features diff: {diff}", "blue")
        cprint(f"compute dino features batched time: {end - beg} seconds", "blue")

        # import pdb; pdb.set_trace()
        # assert np.allclose(all_dino_features, all_dino_features_2)

        all_dino_features = all_dino_features[depth_masks]
    end = time.time()
    
    all_pcd_in_world = np.concatenate(all_pcd_in_world, axis=0)  # (num_cams * num_points, 3)
    all_pcd_in_table_center = np.concatenate(all_pcd_in_table_center, axis=0)

    # if robot_pc is not None:
        ### use open3d to visualize the robot point cloud and the camera point cloud
        # import open3d as o3d
        # o3d_pc_robot = o3d.geometry.PointCloud()
        # o3d_pc_robot.points = o3d.utility.Vector3dVector(robot_pc)
        # o3d_pc_robot.paint_uniform_color([1.0, 0.0, 0.0])

        # o3d_pc_scene = o3d.geometry.PointCloud()
        # o3d_pc_scene.points = o3d.utility.Vector3dVector(all_pcd_in_table_center)
        # o3d_pc_scene.paint_uniform_color([0.0, 1.0, 0.0])
        # o3d.visualization.draw_geometries([o3d_pc_robot, o3d_pc_scene], window_name="before filtering robot points")


        # beg = time.time()
        # non_robot_idx = filter_pcd_a_by_b_radius(all_pcd_in_table_center, robot_pc, thresh=0.02)
        # end = time.time()
        # cprint("filter robot time: {}".format(end - beg), "yellow")
        # all_pcd_in_world = all_pcd_in_world[non_robot_idx]
        # all_pcd_in_table_center = all_pcd_in_table_center[non_robot_idx]
        # if all_cam_rgb_images is not None:
        #     all_rgb_flattend = all_rgb_flattend[non_robot_idx]
        # if use_dino and all_cam_rgb_images is not None:
        #     all_dino_features = all_dino_features[non_robot_idx]


    filter_idx_1 = all_pcd_in_world[:, 1] < 0.4
    filter_idx_2 = all_pcd_in_world[:, 1] > -0.4
    filter_idx_3 = all_pcd_in_world[:, 2] > -0.02
    
    filter_idx = np.logical_and(filter_idx_1, filter_idx_2)
    filter_idx = np.logical_and(filter_idx, filter_idx_3)
    if robot_pc is not None:
        filter_idx_3 = all_pcd_in_table_center[:, 2] > 0
        filter_idx_4 = all_pcd_in_table_center[:, 0] > -0.2
        filter_idx_5 = all_pcd_in_table_center[:, 0] < 0.3
        filter_idx = np.logical_and(filter_idx, filter_idx_3)
        filter_idx = np.logical_and(filter_idx, filter_idx_4)
        filter_idx = np.logical_and(filter_idx, filter_idx_5)

    all_pcd_in_world = all_pcd_in_world[filter_idx]
    all_pcd_in_table_center = all_pcd_in_table_center[filter_idx]
    if all_cam_rgb_images is not None:
        all_rgb_flattend = all_rgb_flattend[filter_idx]
    if use_dino and all_cam_rgb_images is not None:
        all_dino_features = all_dino_features[filter_idx]
    
    beg = time.time()
    all_pcd_in_world, fps_idx = fpsample_pcd(all_pcd_in_world, num_points)
    end = time.time()
    cprint(f"fpsample pcd time: {end - beg} seconds", "blue")
    
    all_pcd_in_table_center = all_pcd_in_table_center[fps_idx]
    if all_cam_rgb_images is not None:
        ### TODO: check the rgb data dtype here
        all_rgb_flattend = all_rgb_flattend[fps_idx]
        # import pdb; pdb.set_trace()
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

    # eef_pos = [cur_eef_pos + np.array([0.0, 0.0, 0.02])]
    # eef_pos = [cur_eef_pos + np.array([0.0, 0.0, 0])]
    # eef_orient_matrix = [cur_eef_matrix]
    # gripper_widths = [cur_gripper_width]
    
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


def get_aloha_future_eef_poses_from_pi05_delta_actions(
    action_chunk,
    eef_pos_robot_base,
    eef_rot_matrix_robot_base,
    eef_gripper_width_franka,
    gripper_delta_scale: float = 1.0,
):
    """Integrate pi05 7D delta actions into absolute EE poses (same frames as low-level 10D path).

    Each row is ``[dx, dy, dz, d_rx, d_ry, d_rz, d_gripper]``: delta position in robot base,
    delta rotation as axis-angle (rotation vector for ``R.from_rotvec``), delta gripper width.

    Rotation matches ``get_aloha_future_eef_poses_from_delta_actions``: right-multiply the current
    orientation by the delta rotation, ``R_new = R_cur @ R_delta`` (see aloha conversion that
    stores 6D delta as a matrix and converts to rotvec for pi05).

    Args:
        action_chunk: (T, 7) or (7,) array/tensor; extra columns are ignored with a warning.
        eef_pos_robot_base: (3,) current EE position in robot base.
        eef_rot_matrix_robot_base: (3, 3) current EE rotation in robot base.
        eef_gripper_width_franka: scalar Franka gripper width (m), same units as 10D path.
        gripper_delta_scale: multiplies delta gripper before integration.

    Returns:
        Same tuple as ``get_aloha_future_eef_poses_from_delta_actions``:
        aloha_world positions, 6D orientations (Aloha convention), Aloha gripper widths, and
        robot-base positions along the chunk (for debugging).
    """
    actions = np.asarray(action_chunk, dtype=np.float64)
    # if actions.ndim == 1:
    #     actions = actions.reshape(1, -1)
    # if actions.shape[-1] < 7:
    #     raise ValueError(
    #         f"pi05 actions need at least 7 dims per step, got shape {actions.shape}"
    #     )
    # if actions.shape[-1] > 7:
    #     cprint(
    #         f"get_aloha_future_eef_poses_from_pi05_delta_actions: truncating action width "
    #         f"{actions.shape[-1]} -> 7",
    #         "yellow",
    #     )
    #     actions = actions[:, :7]

    cur_eef_pos = np.asarray(eef_pos_robot_base, dtype=np.float64).reshape(3).copy()
    cur_eef_matrix = np.asarray(eef_rot_matrix_robot_base, dtype=np.float64).reshape(3, 3).copy()
    cur_gripper_width = float(np.asarray(eef_gripper_width_franka).reshape(()))

    eef_pos = []
    eef_orient_matrix = []
    gripper_widths = []

    for act in actions:
        delta_pos = act[0:3].astype(np.float64)
        # delta_rotvec = act[3:6].astype(np.float64)
        delta_rot_6d = act[3:9].astype(np.float64)
        # delta_gripper = float(act[6]) * float(gripper_delta_scale)
        delta_gripper = float(act[9]) * float(gripper_delta_scale)

        new_eef_pos = cur_eef_pos + delta_pos
        # r_delta = R.from_rotvec(delta_rotvec).as_matrix()
        r_delta = rotation_transfer_6D_to_matrix(delta_rot_6d)
        new_eef_rot_matrix = cur_eef_matrix @ r_delta
        new_gripper_width = cur_gripper_width + delta_gripper
        new_gripper_width = float(np.clip(new_gripper_width, 0.0, 0.04))

        cur_eef_pos = new_eef_pos
        cur_eef_matrix = new_eef_rot_matrix
        cur_gripper_width = new_gripper_width
        eef_pos.append(new_eef_pos)
        eef_orient_matrix.append(new_eef_rot_matrix)
        gripper_widths.append(new_gripper_width)

    eef_pos = np.array(eef_pos).reshape(-1, 3)
    gripper_widths = np.array(gripper_widths)

    aloha_world_eef_pos = transform_from_robot_base_to_table_center(eef_pos)
    aloha_world_eef_orient_matrix = [
        eef_matrix_robot_base_to_aloha_eef_matrix(mat) for mat in eef_orient_matrix
    ]
    aloha_world_eef_orient_6d = [
        rotation_transfer_matrix_to_6D(mat) for mat in aloha_world_eef_orient_matrix
    ]
    aloha_gripper_widths = [x / 0.04 * 80 for x in gripper_widths]
    cprint("gripper width (pi05): {}".format(aloha_gripper_widths), "yellow")

    return aloha_world_eef_pos, aloha_world_eef_orient_6d, aloha_gripper_widths, eef_pos
