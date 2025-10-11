#!/usr/bin/env python3
import numpy as np
from torchvision import transforms
import PIL
import torch
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import matrix_to_rotation_6d
from lerobot.common.policies.high_level.articubot import PointNet2_super, get_weighted_displacement, sample_from_gmm
from lerobot.common.policies.high_level.dino_heatmap import DinoHeatmapNetwork, sample_from_heatmap
from lerobot.common.policies.high_level.dino_3dgp import Dino3DGPNetwork, sample_from_gmm_3dgp, get_weighted_prediction_3dgp
import wandb
from transformers import AutoModel, AutoProcessor
from lerobot.common.utils.aloha_utils import render_aloha_gripper_pcd
import os
from google import genai
from lerobot.common.policies.high_level.classify_utils import setup_client, generate_prompt_for_current_subtask, call_gemini_with_retry, TASK_SPEC, EXAMPLES
from dataclasses import dataclass
from typing import Optional
from PIL import Image

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


@dataclass
class HighLevelConfig:
    """Configuration for HighLevelWrapper"""
    model_type: str = "articubot"  # "articubot", "dino_heatmap", or "dino_3dgp"
    run_id: Optional[str] = None
    max_depth: float = 1.0
    num_points: int = 8192
    in_channels: int = 3
    use_gripper_pcd: bool = False
    use_text_embedding: bool = False
    use_dual_head: bool = False
    use_rgb: bool = False
    use_gemini: bool = False
    is_gmm: bool = False
    dino_model: str = "facebook/dinov2-base"
    intrinsics_txt: str = "lerobot/scripts/aloha_calibration/intrinsics.txt"
    extrinsics_txt: str = "lerobot/scripts/aloha_calibration/T_world_from_camera_est_v6_0709.txt"

    # dino_3dgp specific configs
    use_fourier_pe: bool = False
    fourier_num_frequencies: int = 64
    fourier_include_input: bool = True
    num_transformer_layers: int = 4
    dropout: float = 0.1
    use_source_token: bool = False
    use_gripper_token: bool = True


class HighLevelWrapper:
    def __init__(self, config: HighLevelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_embedding_cache = {}

        if config.use_gemini:
            self.client = setup_client(os.environ.get("RPAD_GEMINI_API_KEY"))
            self.gemini_config = genai.types.GenerateContentConfig(temperature=0.0, candidate_count=1)
            self.model_name = "gemini-2.5-pro"

        self.original_K = np.loadtxt(config.intrinsics_txt) # not scaled
        self.scaled_K = None
        self.cam_to_world = np.loadtxt(config.extrinsics_txt)

        # Initialize model based on type
        if config.model_type == "articubot":
            self.model = initialize_articubot_model(
                config.run_id, config.use_text_embedding, config.use_dual_head,
                config.in_channels, self.device
            )
        elif config.model_type == "dino_heatmap":
            self.model = initialize_dino_heatmap_model(
                config.run_id, config.dino_model, config.use_gripper_pcd,
                config.use_text_embedding, self.device
            )
        elif config.model_type == "dino_3dgp":
            self.model = initialize_dino_3dgp_model(
                config.run_id, config.dino_model, config.use_text_embedding,
                config.use_gripper_token, config.use_source_token, config.use_fourier_pe,
                config.fourier_num_frequencies, config.fourier_include_input,
                config.num_transformer_layers, config.dropout, self.device
            )
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")

        self.rng = np.random.default_rng()
        self.aloha_gripper_idx = torch.tensor([6, 197, 174]) # Handpicked idxs for the aloha

        # For rerun visualization
        self.last_pcd_xyz = None
        self.last_pcd_rgb = None
        self.last_gripper_pcd = None
        self.last_goal_prediction = None

    def _get_gripper_pcd(self, robot_type, robot_kwargs):
        """Extract gripper point cloud for different robot types"""
        if robot_type == "aloha":
            joint_state = robot_kwargs["observation.state"]
            return render_aloha_gripper_pcd(self.cam_to_world, joint_state)
        elif robot_type == "libero_franka":
            from lerobot.common.utils.libero_franka_utils import get_4_points_from_gripper_pos_orient
            return get_4_points_from_gripper_pos_orient(
                gripper_pos=robot_kwargs["ee_pos"],
                gripper_orn=robot_kwargs["ee_quat"],
                cur_joint_angle=robot_kwargs["gripper_angle"],
                world_to_cam_mat=np.linalg.inv(self.cam_to_world),
            )
        else:
            raise NotImplementedError(f"Need to implement code to extract gripper pcd for {robot_type}.")

    def _gripper_pcd_to_token(self, gripper_pcd):
        """
        Convert gripper point cloud (3 points) to gripper token (10-dim).
        Token format: [3 position, 6 rotation (6d), 1 gripper width]

        Args:
            gripper_pcd: (3, 3) numpy array with gripper points

        Returns:
            gripper_token: (10,) numpy array
        """
        # Gripper position (center of first two points - fingertips)
        gripper_pos = (gripper_pcd[0, :] + gripper_pcd[1, :]) / 2

        # Gripper width (distance between fingertips)
        gripper_width = np.linalg.norm(gripper_pcd[0, :] - gripper_pcd[1, :])

        # Gripper orientation from the three points
        # Use palm->center as primary axis
        forward = gripper_pos - gripper_pcd[2, :]
        x_axis = forward / np.linalg.norm(forward)

        # Use finger direction for secondary axis
        finger_vec = gripper_pos - gripper_pcd[0, :]

        # Project finger vector onto plane perpendicular to forward
        finger_projected = finger_vec - np.dot(finger_vec, x_axis) * x_axis
        y_axis = finger_projected / np.linalg.norm(finger_projected)

        # Z completes the frame
        z_axis = np.cross(x_axis, y_axis)

        # Create rotation matrix
        rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=-1)

        # Convert to 6D rotation representation
        rotation_matrix_torch = torch.from_numpy(rotation_matrix).float()
        rotation_6d = matrix_to_rotation_6d(rotation_matrix_torch).numpy()

        # Combine into token
        gripper_token = np.concatenate([gripper_pos, rotation_6d, [gripper_width]])

        return gripper_token

    def _get_text_embedding(self, text, rgb, robot_type, robot_kwargs):
        """Get text embedding with optional Gemini preprocessing and caching"""
        infer_text = self.get_goal_text(text, rgb, robot_type, robot_kwargs)
        if infer_text not in self.text_embedding_cache:
            self.text_embedding_cache[infer_text] = get_siglip_text_embedding(infer_text)
        return self.text_embedding_cache[infer_text]

    def predict(self, text, rgb, depth, robot_type, robot_kwargs):
        if self.config.model_type == "articubot":
            return self._predict_articubot(text, rgb, depth, robot_type, robot_kwargs)
        elif self.config.model_type == "dino_heatmap":
            return self._predict_dino_heatmap(text, rgb, depth, robot_type, robot_kwargs)
        elif self.config.model_type == "dino_3dgp":
            return self._predict_dino_3dgp(text, rgb, depth, robot_type, robot_kwargs)
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def _predict_articubot(self, text, rgb, depth, robot_type, robot_kwargs):
        """Prediction using Articubot point cloud model"""
        if self.scaled_K is None:
            self.scaled_K = get_scaled_intrinsics(self.original_K, (rgb.shape[0], rgb.shape[1]), TARGET_SHAPE)

        pcd = compute_pcd(rgb, depth, self.scaled_K, rgb_preprocess,
                            depth_preprocess, self.device, self.rng,
                            self.config.num_points, self.config.max_depth)
        pcd_xyz, pcd_rgb = pcd[:,:3], pcd[:, 3:]
        # Store for rerun visualization
        self.last_pcd_xyz = pcd_xyz
        self.last_pcd_rgb = pcd_rgb

        if self.config.use_gripper_pcd:
            gripper_pcd = self._get_gripper_pcd(robot_type, robot_kwargs)
            pcd_xyz = concat_gripper_pcd(gripper_pcd, pcd_xyz)
            if self.config.use_rgb:
                gripper_rgb = np.zeros((gripper_pcd.shape[0], 3))
                pcd_rgb = np.concatenate([gripper_rgb, pcd_rgb], axis=0)
            self.last_gripper_pcd = gripper_pcd

        if self.config.use_rgb:
            pcd = np.concatenate([pcd_xyz, pcd_rgb], axis=1)
        else:
            pcd = pcd_xyz

        text_embed = self._get_text_embedding(text, rgb, robot_type, robot_kwargs)

        # Run inference
        goal_prediction = inference_articubot(self.model, pcd, text_embed, self.config.is_gmm, self.device)
        # Store for rerun visualization
        self.last_goal_prediction = goal_prediction

        return goal_prediction

    def _predict_dino_heatmap(self, text, rgb, depth, robot_type, robot_kwargs):
        """Prediction using DINO heatmap model"""
        # Get gripper point cloud if needed
        gripper_pcd = None
        if self.config.use_gripper_pcd:
            gripper_pcd = self._get_gripper_pcd(robot_type, robot_kwargs)
            self.last_gripper_pcd = gripper_pcd

        # Get text embedding if needed
        text_embed = None
        if self.config.use_text_embedding:
            text_embed = self._get_text_embedding(text, rgb, robot_type, robot_kwargs)

        # Run inference - returns sampled 2D coord in 224x224 space
        coord_2d = inference_dino_heatmap(self.model, rgb, gripper_pcd, text_embed, self.device)

        return coord_2d

    def _predict_dino_3dgp(self, text, rgb, depth, robot_type, robot_kwargs):
        """Prediction using DINO 3D Goal Prediction model"""
        if self.scaled_K is None:
            self.scaled_K = get_scaled_intrinsics(self.original_K, (rgb.shape[0], rgb.shape[1]), TARGET_SHAPE)

        # Just for visualization, keep only 500 points
        pcd = compute_pcd(rgb, depth, self.scaled_K, rgb_preprocess,
                         depth_preprocess, self.device, self.rng,
                         500, self.config.max_depth)
        pcd_xyz, pcd_rgb = pcd[:, :3], pcd[:, 3:]
        # Store for rerun visualization
        self.last_pcd_xyz = pcd_xyz
        self.last_pcd_rgb = pcd_rgb

        # Get gripper point cloud if needed
        gripper_token = None
        if self.config.use_gripper_token:
            gripper_pcd = self._get_gripper_pcd(robot_type, robot_kwargs)
            self.last_gripper_pcd = gripper_pcd
            gripper_token = self._gripper_pcd_to_token(gripper_pcd[self.aloha_gripper_idx])

        # Get text embedding if needed
        text_embed = None
        if self.config.use_text_embedding:
            text_embed = self._get_text_embedding(text, rgb, robot_type, robot_kwargs)

        goal_prediction = inference_dino_3dgp(
            self.model, rgb, depth, self.scaled_K, gripper_token, text_embed,
            robot_type, self.config.is_gmm, self.config.max_depth, self.device
        )

        # Store for rerun visualization
        self.last_goal_prediction = goal_prediction

        return goal_prediction

    def get_goal_text(self, text, rgb, robot_type, robot_kwargs):
        if not self.config.use_gemini:
            return text

        if robot_type == "aloha":
            joint_state = robot_kwargs["observation.state"]
        else:
            raise NotImplementedError(f"Need to implement code to extract joint state for {robot_type}.")

        pil_image = PIL.Image.fromarray(rgb)

        subgoals = TASK_SPEC[text]
        GRIPPER_MIN, GRIPPER_MAX = 0.0, 0.041
        gripper_state = joint_state[7]
        gripper_state_scaled = (gripper_state - GRIPPER_MIN) / GRIPPER_MAX

        prompt = generate_prompt_for_current_subtask(
                text, subgoals, pil_image, gripper_state_scaled, EXAMPLES
        )
        goal_text = call_gemini_with_retry(self.client, self.model_name, prompt, self.gemini_config)
        return goal_text

    def project(self, goal_prediction, img_shape, goal_repr="heatmap"):
        """Project goal prediction to image space"""
        if self.config.model_type == "articubot" or self.config.model_type == "dino_3dgp":
            return self._project_articubot(goal_prediction, img_shape, goal_repr)
        elif self.config.model_type == "dino_heatmap":
            return self._compute_dino_heatmap(goal_prediction, img_shape)
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def _project_articubot(self, goal_prediction, img_shape, goal_repr="heatmap"):
        """Project 3D points to 2D image space for Articubot"""
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

    def _compute_dino_heatmap(self, coord_2d_224, img_shape):
        """Scale 2D coord from 224x224 to full image space and create distance-based heatmap"""
        # coord_2d_224 is (2,) [x, y] in 224x224 space
        # Scale to original image space (img_shape is (H, W, ...))
        H, W = img_shape[0], img_shape[1]

        # Inverse of the preprocessing: Resize (aspect-ratio preserving) + CenterCrop
        # Step 1: Resize scales the shorter edge to 224
        scale_factor = TARGET_SHAPE / min(H, W)
        # After resize: (H * scale_factor, W * scale_factor)

        # Step 2: CenterCrop takes 224x224 from center
        # The crop offsets in the resized space
        crop_offset_x = (W * scale_factor - TARGET_SHAPE) / 2
        crop_offset_y = (H * scale_factor - TARGET_SHAPE) / 2

        # Inverse transform: add back crop offset, then scale back to original size
        coord_2d_resized = np.array([
            coord_2d_224[0] + crop_offset_x,
            coord_2d_224[1] + crop_offset_y
        ])

        coord_2d_full = coord_2d_resized / scale_factor
        coord_2d_full = coord_2d_full.astype(int)

        # Clip to image bounds
        coord_2d_full = np.clip(coord_2d_full, [0, 0], [W - 1, H - 1])

        # Create single-channel distance-based heatmap using same logic as articubot
        max_distance = np.sqrt(W**2 + H**2)
        y_coords, x_coords = np.mgrid[0:H, 0:W]
        pixel_coords = np.stack([x_coords, y_coords], axis=-1)

        # Compute distances from target point
        distances = np.linalg.norm(pixel_coords - coord_2d_full, axis=-1)  # (H, W)

        # Apply square root transformation for steeper near-target gradients
        heatmap = (np.sqrt(distances / max_distance) * 255)
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)

        # Return 3-channel heatmap (repeat single channel)
        return np.stack([heatmap, heatmap, heatmap], axis=-1)  # (H, W, 3)

    def predict_and_project(self, text, rgb, depth, robot_type, robot_kwargs):
        goal_prediction = self.predict(text, rgb, depth, robot_type, robot_kwargs)
        goal_gripper_proj = self.project(goal_prediction, rgb.shape)
        return goal_gripper_proj


def initialize_articubot_model(run_id, use_text_embedding, use_dual_head, in_channels, device):
    """Initialize Articubot PointNet2 model from wandb artifact"""
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

    model = PointNet2_super(num_classes=13, input_channel=in_channels, use_text_embedding=use_text_embedding, use_dual_head=use_dual_head)
    model.load_state_dict(state_dict)

    model = model.eval()
    model = model.to(device)

    return model

def initialize_dino_heatmap_model(run_id, dino_model, use_gripper_pcd, use_text_embedding, device):
    """Initialize DINO heatmap model from wandb artifact"""

    # Simple config object to match what DinoHeatmapNetwork expects
    class ModelConfig:
        def __init__(self, dino_model, use_gripper_pcd, use_text_embedding):
            self.dino_model = dino_model
            self.use_gripper_pcd = use_gripper_pcd
            self.use_text_embedding = use_text_embedding

    model_cfg = ModelConfig(dino_model, use_gripper_pcd, use_text_embedding)
    model = DinoHeatmapNetwork(model_cfg)

    artifact_dir = "wandb"
    checkpoint_reference = f"r-pad/lfd3d/best_pix_dist_model-{run_id}:best"
    api = wandb.Api()
    artifact = api.artifact(checkpoint_reference, type="model")
    ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    ckpt = torch.load(ckpt_file)
    # Remove the "network." prefix, since we're not using Lightning here.
    state_dict = {k.replace("network.",""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)

    model = model.eval()
    model = model.to(device)

    return model

def initialize_dino_3dgp_model(
    run_id, dino_model, use_text_embedding, use_gripper_token, use_source_token,
    use_fourier_pe, fourier_num_frequencies, fourier_include_input,
    num_transformer_layers, dropout, device
):
    """Initialize DINO 3D Goal Prediction model from wandb artifact"""

    # Simple config object to match what Dino3DGPNetwork expects
    class ModelConfig:
        def __init__(self, dino_model, use_text_embedding, use_gripper_token,
                     use_source_token, use_fourier_pe, fourier_num_frequencies,
                     fourier_include_input, num_transformer_layers, dropout):
            self.dino_model = dino_model
            self.use_text_embedding = use_text_embedding
            self.use_gripper_token = use_gripper_token
            self.use_source_token = use_source_token
            self.use_fourier_pe = use_fourier_pe
            self.fourier_num_frequencies = fourier_num_frequencies
            self.fourier_include_input = fourier_include_input
            self.num_transformer_layers = num_transformer_layers
            self.dropout = dropout

    model_cfg = ModelConfig(
        dino_model, use_text_embedding, use_gripper_token,
        use_source_token, use_fourier_pe, fourier_num_frequencies,
        fourier_include_input, num_transformer_layers, dropout
    )
    model = Dino3DGPNetwork(model_cfg)

    artifact_dir = "wandb"
    checkpoint_reference = f"r-pad/lfd3d/best_rmse_model-{run_id}:best"
    api = wandb.Api()
    artifact = api.artifact(checkpoint_reference, type="model")
    ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    ckpt = torch.load(ckpt_file)
    # Remove the "network." prefix, since we're not using Lightning here.
    state_dict = {k.replace("network.",""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)

    model = model.eval()
    model = model.to(device)

    return model

def inference_articubot(model, pcd, text_embedding, is_gmm, device):
    """
    Run Articubot model inference on point cloud data.

    Args:
        model (PointNet2_super): Trained model.
        pcd (np.ndarray): Point cloud coordinates (N, K) or batched.
        text_embedding (np.ndarray): Goal text embedding.
        is_gmm (bool): Whether to use GMM sampling or weighted displacement.
        device (torch.device): Device for inference.

    Returns:
        np.ndarray: Predicted goal displacement (e.g., 4x3 array).
    """
    with torch.no_grad():
        if len(pcd.shape) == 2:
            pcd = pcd.transpose(1,0)[None] # [1, K, N]
        elif len(pcd.shape) == 3: # batched inference
            pcd = pcd.transpose(0,2,1)
        pcd = torch.from_numpy(pcd.astype(np.float32)).to(device)
        text_embedding = torch.from_numpy(text_embedding.astype(np.float32)[None]).to(device)
        outputs = model(pcd, text_embedding, data_source=["robot"]) # [1, N, 13]
        if not is_gmm:
            goal_prediction = get_weighted_displacement(pcd.permute(0,2,1), outputs).squeeze().cpu().numpy() # [4, 3]
        else:
            viz_gmm = False
            if viz_gmm:
                # Visualize GMM predictions in rerun
                visualize_gmm_predictions(pcd, outputs)
            goal_prediction = sample_from_gmm(pcd.permute(0,2,1), outputs).squeeze().cpu().numpy() # [4, 3]
        return goal_prediction

def inference_dino_heatmap(model, rgb, gripper_pcd, text_embedding, device):
    """
    Run DINO heatmap model inference on RGB image and sample a 2D coordinate.

    Args:
        model (DinoHeatmapNetwork): Trained model.
        rgb (np.ndarray): RGB image (H, W, 3), uint8.
        gripper_pcd (np.ndarray): Optional gripper point cloud (N, 3).
        text_embedding (np.ndarray): Optional text embedding.
        device (torch.device): Device for inference.

    Returns:
        np.ndarray: Sampled 2D coordinate in 224x224 space (2,) [x, y].
    """

    rgb_ = np.asarray(rgb_preprocess(Image.fromarray(rgb))).copy()
    rgb_ = torch.from_numpy(rgb_).unsqueeze(0).permute(0,3,1,2)

    with torch.no_grad():
        # Convert gripper_pcd to torch if provided
        if gripper_pcd is not None:
            gripper_pcd = torch.from_numpy(gripper_pcd.astype(np.float32)).unsqueeze(0).to(device)  # (1, N, 3)

        # Convert text_embedding to torch if provided
        text_embed = None
        if text_embedding is not None:
            text_embed = torch.from_numpy(text_embedding.astype(np.float32)).unsqueeze(0).to(device)  # (1, D)

        score_map = model(
            image=rgb_,
            gripper_pcd=gripper_pcd,
            text_embedding=text_embed
        )  # (1, 1, 224, 224)

        # Sample a 2D coordinate from the score_map distribution
        sampled_coord = sample_from_heatmap(score_map)  # (1, 2) [x, y]

        return sampled_coord.squeeze(0).cpu().numpy()  # (2,) [x, y]

def inference_dino_3dgp(model, rgb, depth, intrinsics, gripper_token, text_embedding,
                        robot_type, is_gmm, max_depth, device):
    """
    Run DINO 3D Goal Prediction model inference on RGB+depth and predict 3D goal points.

    Args:
        model (Dino3DGPNetwork): Trained model.
        rgb (np.ndarray): RGB image (H, W, 3), uint8.
        depth (np.ndarray): Depth image (H, W), uint16 in mm.
        intrinsics (np.ndarray): Camera intrinsics (3, 3), scaled to 224x224.
        gripper_token (np.ndarray): Optional gripper token (10,).
        text_embedding (np.ndarray): Optional text embedding (1152,).
        robot_type (str): Robot type (e.g., "aloha", "robot").
        is_gmm (bool): Whether to use GMM sampling or weighted average.
        max_depth (float): Maximum depth threshold in meters.
        device (torch.device): Device for inference.

    Returns:
        pred_points: (4, 3) predicted 3D goal points
    """
    # Preprocess RGB
    rgb_ = np.asarray(rgb_preprocess(Image.fromarray(rgb))).copy()
    rgb_ = torch.from_numpy(rgb_).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)  # (1, 3, 224, 224)

    # Preprocess depth
    depth_ = (depth / 1000.0).squeeze().astype(np.float32)  # Convert mm to meters
    depth_ = PIL.Image.fromarray(depth_)
    depth_ = np.asarray(depth_preprocess(depth_)).copy()
    depth_[depth_ > max_depth] = 0  # Mask out far depths
    depth_ = torch.from_numpy(depth_).unsqueeze(0).float().to(device)  # (1, 224, 224)

    # Intrinsics to torch
    intrinsics_ = torch.from_numpy(intrinsics.astype(np.float32)).unsqueeze(0).to(device)  # (1, 3, 3)

    with torch.no_grad():
        # Convert gripper_token to torch if provided
        gripper_tok = None
        if gripper_token is not None:
            gripper_tok = torch.from_numpy(gripper_token.astype(np.float32)).unsqueeze(0).to(device)  # (1, 10)

        # Convert text_embedding to torch if provided
        text_embed = None
        if text_embedding is not None:
            text_embed = torch.from_numpy(text_embedding.astype(np.float32)).unsqueeze(0).to(device)  # (1, 1152)

        # Determine source
        source = [robot_type] if model.use_source_token else None

        # Forward through network
        outputs, patch_coords = model(
            image=rgb_,
            depth=depth_,
            intrinsics=intrinsics_,
            gripper_token=gripper_tok,
            text_embedding=text_embed,
            source=source
        )  # outputs: (1, 256, 13), patch_coords: (1, 256, 3)

        # Sample or get weighted prediction
        if is_gmm:
            # Visualize GMM predictions
            viz_gmm = False
            if viz_gmm:
                visualize_gmm_predictions(patch_coords, outputs)
            pred_points = sample_from_gmm_3dgp(outputs, patch_coords)  # (1, 4, 3)
        else:
            pred_points = get_weighted_prediction_3dgp(outputs, patch_coords)  # (1, 4, 3)

        return pred_points.squeeze(0).cpu().numpy()  # (4, 3)

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
    rgb_ = ((np.asarray(rgb_preprocess(rgb_), dtype=np.float32) * 2 / 255.) - 1)

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

def visualize_gmm_predictions(pcd, outputs):
    """
    Visualize GMM predictions in rerun.

    Args:
        pcd: torch.Tensor of shape [1, 7, N] where first 3 channels are xyz
        outputs: torch.Tensor of shape [1, N, 13] containing 4 xyz vectors and 1 weight per point
    """
    import rerun as rr
    # Extract point cloud xyz and convert to numpy
    pcd_xyz = pcd[0, :3, :].permute(1, 0).cpu().numpy()  # [N, 3]

    # Extract outputs: 4 displacement vectors (12 channels) + 1 weight (1 channel)
    outputs_np = outputs[0].cpu().numpy()  # [N, 13]
    displacements = outputs_np[:, :12].reshape(-1, 4, 3)  # [N, 4, 3]
    weights = outputs_np[:, 12]  # [N]

    # Softmax the weights over all points
    weights_exp = np.exp(weights - np.max(weights))
    weights_softmax = weights_exp / weights_exp.sum()  # [N]

    # Random sample 1000 points for visualization
    num_points = len(pcd_xyz)
    sample_size = min(500, num_points)
    sample_indices = np.random.choice(num_points, sample_size, replace=False)

    pcd_xyz_sampled = pcd_xyz[sample_indices]
    displacements_sampled = displacements[sample_indices]
    weights_softmax_sampled = weights_softmax[sample_indices]

    # Log original point cloud (sampled)
    rr.log("gmm/point_cloud", rr.Points3D(pcd_xyz_sampled, radii=0.003))

    # Show only one displacement vector per point (first one for simplicity)
    vec_idx = 0
    target_points = pcd_xyz_sampled + displacements_sampled[:, vec_idx, :]  # [sample_size, 3]

    # Create line segments from each point to its target
    positions = np.zeros((sample_size, 2, 3))
    positions[:, 0, :] = pcd_xyz_sampled  # Start points
    positions[:, 1, :] = target_points  # End points

    # Create color map based on softmax weights
    # Use log scale for better visualization since weights can be very skewed
    weights_log = np.log(weights_softmax_sampled + 1e-10)
    w_min = weights_log.min()
    w_max = weights_log.max()
    w_range = w_max - w_min
    if w_range > 0:
        weights_norm = (weights_log - w_min) / w_range
    else:
        weights_norm = np.ones_like(weights_log) * 0.5

    colors = np.zeros((sample_size, 3), dtype=np.uint8)
    for i, w in enumerate(weights_norm):
        # Colormap: blue (low) -> cyan -> green -> yellow -> red (high)
        if w < 0.25:
            # Blue to cyan
            t = w * 4
            colors[i] = [0, int(t * 255), 255]
        elif w < 0.5:
            # Cyan to green
            t = (w - 0.25) * 4
            colors[i] = [0, 255, int((1 - t) * 255)]
        elif w < 0.75:
            # Green to yellow
            t = (w - 0.5) * 4
            colors[i] = [int(t * 255), 255, 0]
        else:
            # Yellow to red
            t = (w - 0.75) * 4
            colors[i] = [255, int((1 - t) * 255), 0]

    # Log line strips
    rr.log(
        "gmm/displacement_vectors",
        rr.LineStrips3D(
            positions,  # [sample_size, 2, 3]
            colors=colors,
            radii=0.002
        )
    )
