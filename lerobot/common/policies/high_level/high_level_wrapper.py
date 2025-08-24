#!/usr/bin/env python3
import numpy as np
from torchvision import transforms
import PIL
import torch
from pytorch3d.ops import sample_farthest_points
from lerobot.common.policies.high_level.articubot import PointNet2_super, get_weighted_displacement, sample_from_gmm
import wandb
from transformers import AutoModel, AutoProcessor
from lerobot.common.utils.aloha_utils import render_aloha_gripper_pcd
from lerobot.common.utils.libero_franka_utils import get_4_points_from_gripper_pos_orient
import os
from google import genai
from lerobot.common.policies.high_level.classify_utils import setup_client, generate_prompt_for_current_subtask, call_gemini_with_retry, TASK_SPEC, EXAMPLES


TARGET_SHAPE = 224
rgb_preprocess = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SHAPE,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(TARGET_SHAPE),
    ]
)
depth_preprocess = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SHAPE,
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.CenterCrop(TARGET_SHAPE),
    ]
)

class HighLevelWrapper:
    def __init__(self,
                 run_id,
                 max_depth=1.0,
                 num_points=8192,
                 in_channels=3,
                 use_gripper_pcd=False,
                 use_text_embedding=False,
                 text=None,
                 use_gemini=False,
                 is_gmm=False,
                 intrinsics_txt=None,
                 extrinsics_txt=None):
        super().__init__()

        #############################################
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_depth = max_depth
        self.num_points = num_points
        self.is_gmm = is_gmm
        self.use_text_embedding = use_text_embedding
        self.use_gripper_pcd = use_gripper_pcd
        self.text_embedding = {}
        self.text = text

        self.use_gemini = use_gemini
        if self.use_gemini:
            self.client = setup_client(os.environ.get("RPAD_GEMINI_API_KEY"))
            self.gemini_config = genai.types.GenerateContentConfig(temperature=0.0, candidate_count=1)
            self.model_name = "gemini-2.5-pro"
            self.subgoals = TASK_SPEC[text]
            for subgoal in self.subgoals:
                self.text_embedding[subgoal] = get_siglip_text_embedding(subgoal)
        else:
            self.text_embedding[self.text] = get_siglip_text_embedding(self.text)


        self.original_K = np.loadtxt(intrinsics_txt) # not scaled
        self.scaled_K = None
        self.cam_to_world = np.loadtxt(extrinsics_txt)
        #############################################

        self.model = initialize_model(run_id, use_text_embedding, in_channels, self.device)
        self.rng = np.random.default_rng()

    def predict(self, rgb, depth, robot_type, robot_kwargs):
        if self.scaled_K is None:
            self.scaled_K = get_scaled_intrinsics(self.original_K, (rgb.shape[0], rgb.shape[1]), TARGET_SHAPE)

        pcd = compute_pcd(rgb, depth, self.scaled_K, rgb_preprocess,
                            depth_preprocess, self.device, self.rng,
                            self.num_points, self.max_depth)
        pcd_xyz = pcd[:,:3]

        if self.use_gripper_pcd:
            if robot_type == "aloha":
                joint_state = robot_kwargs["observation.state"]
                gripper_pcd = render_aloha_gripper_pcd(self.cam_to_world, joint_state)
            elif robot_type == "libero_franka":
                gripper_pcd = get_4_points_from_gripper_pos_orient(
                            gripper_pos=robot_kwargs["ee_pos"],
                            gripper_orn=robot_kwargs["ee_quat"],
                            cur_joint_angle=robot_kwargs["gripper_angle"],
                            world_to_cam_mat=np.linalg.inv(self.cam_to_world),
                        )
            else:
                raise NotImplementedError(f"Need to implement code to extract gripper pcd for {robot_type}.")
            pcd_xyz = concat_gripper_pcd(gripper_pcd, pcd_xyz)

        infer_text = self.get_goal_text(rgb, robot_type, robot_kwargs)
        #### Run inference
        goal_prediction = inference(self.model, pcd_xyz, self.text_embedding[infer_text], self.is_gmm, self.device)

        return goal_prediction

    def get_goal_text(self, rgb, robot_type, robot_kwargs):
        if not self.use_gemini:
            return self.text

        if robot_type == "aloha":
            joint_state = robot_kwargs["observation.state"]
        else:
            raise NotImplementedError(f"Need to implement code to extract joint state for {robot_type}.")

        pil_image = PIL.Image.fromarray(rgb)

        GRIPPER_MIN, GRIPPER_MAX = 0.0, 0.041
        gripper_state = joint_state[7]
        gripper_state_scaled = (gripper_state - GRIPPER_MIN) / GRIPPER_MAX
        
        prompt = generate_prompt_for_current_subtask(
                self.text, self.subgoals, pil_image, gripper_state_scaled, EXAMPLES
        )
        goal_text = call_gemini_with_retry(self.client, self.model_name, prompt, self.gemini_config)
        return goal_text

    def project(self, goal_prediction, img_shape, goal_repr="heatmap"):
        assert goal_repr in ["mask", "heatmap"]
        urdf_proj_hom = (self.original_K @ goal_prediction.T).T
        urdf_proj = (urdf_proj_hom / urdf_proj_hom[:, 2:])[:, :2]
        urdf_proj = np.clip(urdf_proj, [0, 0], [img_shape[1] - 1, img_shape[0] - 1]).astype(int)
        goal_gripper_proj = np.zeros((img_shape[0], img_shape[1], 3))
        if goal_repr == "mask":
            goal_gripper_proj[urdf_proj[:, 1], urdf_proj[:, 0]] = 255
        elif goal_repr == "heatmap":
            max_distance = np.sqrt(img_shape[1]**2 + img_shape[0]**2)
            y_coords, x_coords = np.mgrid[0:img_shape[0], 0:img_shape[1]]
            pixel_coords = np.stack([x_coords, y_coords], axis=-1)
            for i in range(3):
                target_point = urdf_proj[i]  # (2,)
                distances = np.linalg.norm(pixel_coords - target_point, axis=-1)  # (height, width)
                goal_gripper_proj[:, :, i] = distances

            # Apply square root transformation for steeper near-target gradients
            goal_gripper_proj = (np.sqrt(goal_gripper_proj / max_distance) * 255)
            goal_gripper_proj = np.clip(goal_gripper_proj, 0, 255)
        return goal_gripper_proj.astype(np.uint8)

    def predict_and_project(self, rgb, depth, robot_type, robot_kwargs):
        goal_prediction = self.predict(rgb, depth, robot_type, robot_kwargs)
        goal_gripper_proj = self.project(goal_prediction, rgb.shape)
        return goal_gripper_proj


def initialize_model(run_id, use_text_embedding, in_channels, device):
    # Initialize WandB API and download artifact
    # Follows naming convention in lfd3d
    artifact_dir = "wandb"
    checkpoint_reference = f"r-pad/lfd3d/best_rmse_model-{run_id}:best"
    api = wandb.Api()
    artifact = api.artifact(checkpoint_reference, type="model")
    ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    ckpt = torch.load(ckpt_file)
    # Remove the "network." prefix, since we're not using Lightning here.
    state_dict = {k.replace("network.",""): v for k, v in ckpt["state_dict"].items()}

    model = PointNet2_super(num_classes=13, input_channel=in_channels, use_text_embedding=use_text_embedding)
    model.load_state_dict(state_dict)

    model = model.eval()
    model = model.to(device)

    return model

def inference(model, pcd_xyz, text_embedding, is_gmm, device):
    """
    Run model inference on point cloud data.

    Args:
        model (PointNet2_super): Trained model.
        pcd_xyz (np.ndarray): Point cloud coordinates (N, 3) or batched.
        text_embedding (np.ndarray): Goal text embedding.
        device (torch.device): Device for inference.

    Returns:
        np.ndarray: Predicted goal displacement (e.g., 4x3 array).
    """
    with torch.no_grad():
        if len(pcd_xyz.shape) == 2:
            pcd_xyz = pcd_xyz.transpose(1,0)[None] # [1, 3, N]
        elif len(pcd_xyz.shape) == 3: # batched inference
            pcd_xyz = pcd_xyz.transpose(0,2,1)
        pcd_xyz = torch.from_numpy(pcd_xyz.astype(np.float32)).to(device)
        text_embedding = torch.from_numpy(text_embedding.astype(np.float32)[None]).to(device)
        outputs = model(pcd_xyz, text_embedding) # [1, N, 13]
        if not is_gmm:
            goal_prediction = get_weighted_displacement(pcd_xyz.permute(0,2,1), outputs).squeeze().cpu().numpy() # [4, 3]
        else:
            goal_prediction = sample_from_gmm(pcd_xyz.permute(0,2,1), outputs).squeeze().cpu().numpy() # [4, 3]
        return goal_prediction

def compute_pcd(rgb, depth, K, rgb_preprocess, depth_preprocess, device, rng, num_points, max_depth):
    """
    Compute a downsampled point cloud from RGB and depth images.

    Args:
        rgb (np.ndarray): RGB image array (H, W, 3). np.uint8
        depth (np.ndarray): Depth image array (H, W). np.uint16
        K (np.ndarray): 3x3 camera intrinsic matrix.
        rgb_preprocess (transforms.Compose): Preprocessing for RGB.
        depth_preprocess (transforms.Compose): Preprocessing for depth.
        device (torch.device): Device for computations.
        rng (np.random.Generator): Random number generator.
        num_points (int): Number of points to sample.
        max_depth (float): Maximum depth threshold.

    Returns:
        np.ndarray: Downsampled point cloud (N, 6) with XYZ and RGB.
    """
    # Downsample images
    rgb_ = PIL.Image.fromarray(rgb)
    rgb_ = np.asarray(rgb_preprocess(rgb_))

    depth_ = (depth / 1000.0).squeeze().astype(np.float32)
    depth_ = PIL.Image.fromarray(depth_)
    depth_ = np.asarray(depth_preprocess(depth_))

    height, width = depth_.shape
    # Create pixel coordinate grid
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Flatten grid coordinates and depth
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = depth_.flatten()
    rgb_flat = rgb_.reshape(-1, 3)

    # Remove points with invalid depth
    valid_depth = np.logical_and(z_flat > 0, z_flat < max_depth)
    x_flat = x_flat[valid_depth]
    y_flat = y_flat[valid_depth]
    z_flat = z_flat[valid_depth]
    rgb_flat = rgb_flat[valid_depth]

    # Create homogeneous pixel coordinates
    pixels = np.stack([x_flat, y_flat, np.ones_like(x_flat)], axis=0)

    # Unproject points using K inverse
    K_inv = np.linalg.inv(K)
    points = K_inv @ pixels
    points = points * z_flat
    points = points.T  # Shape: (N, 3)

    scene_pcd_pt3d = torch.from_numpy(points)
    scene_pcd_downsample, scene_points_idx = sample_farthest_points(
        scene_pcd_pt3d[None], K=num_points, random_start_point=False
    )
    scene_pcd = scene_pcd_downsample.squeeze().numpy()

    # Get corresponding colors at the indices
    scene_rgb_pcd = rgb_flat[scene_points_idx.squeeze().numpy()]

    pcd = np.concatenate([scene_pcd, scene_rgb_pcd], axis=1)
    return pcd

def concat_gripper_pcd(gripper_pcd, pcd_xyz):
    """
    Concatenate gripper point cloud with scene point cloud.

    Args:
        gripper_pcd (np.ndarray): Gripper points (M, 3).
        pcd_xyz (np.ndarray): Scene points (N, 3).

    Returns:
        np.ndarray: Combined points (M+N, 4)
    """
    gripper_pcd = np.concatenate(
        [
            gripper_pcd,
            np.ones((gripper_pcd.shape[0], 1)),
        ],
        axis=1,
    )
    pcd_xyz = np.concatenate(
        [
            pcd_xyz,
            np.zeros((pcd_xyz.shape[0], 1)),
        ],
        axis=1,
    )
    pcd_xyz = np.concatenate([gripper_pcd, pcd_xyz], axis=0)
    return pcd_xyz

def get_scaled_intrinsics(K, orig_shape, target_shape):
    """
    Scale camera intrinsics based on image resizing and cropping.

    Args:
        K (np.ndarray): Original 3x3 camera intrinsic matrix.
        orig_shape (tuple): Original image shape (height, width).
        target_shape (int): Target size for resize and crop.

    Returns:
        np.ndarray: Scaled 3x3 intrinsic matrix.
    """
    # Getting scale factor from torchvision.transforms.Resize behaviour
    K_ = K.copy()

    scale_factor = target_shape / min(orig_shape)

    # Apply the scale factor to the intrinsics
    K_[0, 0] *= scale_factor  # fx
    K_[1, 1] *= scale_factor  # fy
    K_[0, 2] *= scale_factor  # cx
    K_[1, 2] *= scale_factor  # cy

    # Adjust the principal point (cx, cy) for the center crop
    crop_offset_x = (orig_shape[1] * scale_factor - target_shape) / 2
    crop_offset_y = (orig_shape[0] * scale_factor - target_shape) / 2

    # Adjust the principal point (cx, cy) for the center crop
    K_[0, 2] -= crop_offset_x  # Adjust cx for crop
    K_[1, 2] -= crop_offset_y  # Adjust cy for crop
    return K_

def get_siglip_text_embedding(
    caption, siglip=None, siglip_processor=None, device="cuda"
):
    if siglip is None or siglip_processor is None:
        siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(
            device
        )
        siglip_processor = AutoProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )

    # Process text input
    inputs = siglip_processor(text=[caption], return_tensors="pt", padding=True, truncation=True,  max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate embeddings
    with torch.no_grad():
        text_embedding = siglip.get_text_features(**inputs)

    return text_embedding.cpu().squeeze().numpy()

