#!/usr/bin/env python3
import numpy as np
from torchvision import transforms
import PIL
import torch
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from lerobot.common.policies.high_level.articubot import PointNet2_super, get_weighted_displacement, sample_from_gmm
from lerobot.common.policies.high_level.dino_heatmap import DinoHeatmapNetwork, sample_from_heatmap
from lerobot.common.policies.high_level.dino_3dgp import Dino3DGPNetwork, sample_from_gmm_3dgp, get_weighted_prediction_3dgp
import wandb
from transformers import AutoModel, AutoProcessor
from lerobot.common.utils.aloha_utils import render_aloha_gripper_pcd, render_aloha_gripper_mesh
import os
from google import genai
from lerobot.common.policies.high_level.classify_utils import setup_client, generate_prompt_for_current_subtask, call_gemini_with_retry, TASK_SPEC, EXAMPLES
from dataclasses import dataclass, field
from typing import Optional, Dict
from PIL import Image
import json
from lerobot.scripts.dataset_utils import generate_heatmap_from_points
import types
from torch import nn, optim
import torch.distributions as D
import torch.nn.functional as F

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

def _gripper_pcd_to_token(gripper_pcd):
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

def _get_gripper_pcd(robot_type, robot_kwargs):
    """Extract gripper point cloud for different robot types"""
    if robot_type == "aloha":
        joint_state = robot_kwargs["observation.state"]
        return render_aloha_gripper_pcd(np.eye(4), joint_state) # render in world frame
    elif robot_type == "libero_franka":
        from lerobot.common.utils.libero_franka_utils import get_4_points_from_gripper_pos_orient
        return get_4_points_from_gripper_pos_orient(
            gripper_pos=robot_kwargs["ee_pos"],
            gripper_orn=robot_kwargs["ee_quat"],
            cur_joint_angle=robot_kwargs["gripper_angle"],
            world_to_cam_mat=np.eye(4), # render in world frame
        )#[self.GRIPPER_IDX[robot_type]]
    else:
        raise NotImplementedError(f"Need to implement code to extract gripper pcd for {robot_type}.")

@dataclass
class HighLevelConfig:
    """Configuration for HighLevelWrapper"""
    model_type: str = "articubot"  # "articubot", "dino_heatmap", or "dino_3dgp"
    run_id: Optional[str] = None
    entity: str = "r-pad"
    project: str = "lfd3d"
    checkpoint_type: str = "rmse"
    max_depth: float = 1.5
    num_points: int = 8192
    in_channels: int = 3
    use_gripper_pcd: bool = False
    use_text_embedding: bool = False
    use_dual_head: bool = False
    use_rgb: bool = False
    use_gemini: bool = False
    is_gmm: bool = False
    dino_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    calibration_data: Dict[str, Dict] = field(default_factory=dict)

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

        self.calibration_data = config.calibration_data

        # Extract camera configurations
        self.camera_names = list(self.calibration_data.keys())
        self.num_cameras = len(self.camera_names)

        # Load intrinsics and extrinsics for all cameras
        self.original_Ks = []  # List of intrinsics matrices
        self.cam_to_worlds = []  # List of extrinsics matrices (T_world_from_camera)

        for cam_name in self.camera_names:
            cam_config = self.calibration_data[cam_name]
            intrinsics_path = os.path.join("lerobot/scripts/", cam_config["intrinsics"])
            extrinsics_path = os.path.join("lerobot/scripts/", cam_config["extrinsics"])

            self.original_Ks.append(np.loadtxt(intrinsics_path))
            self.cam_to_worlds.append(np.loadtxt(extrinsics_path))

        self.scaled_Ks = [None] * self.num_cameras  # Will be set during inference

        # Initialize model based on type
        if config.model_type == "articubot":
            self.model = initialize_articubot_model(
                config.run_id, config.use_text_embedding, config.use_dual_head,
                config.in_channels, self.device
            )
        elif config.model_type == "dino_heatmap":
            if self.num_cameras != 1:
                raise NotImplementedError("dino_heatmap model type is not yet supported for multiview inference")
            self.model = initialize_dino_heatmap_model(config.entity, config.project, config.checkpoint_type,
                config.run_id, config.dino_model, config.use_gripper_pcd,
                config.use_text_embedding, self.device
            )
        elif config.model_type == "dino_3dgp":
            self.model = initialize_dino_3dgp_model(config.entity, config.project, config.checkpoint_type,
                config.run_id, config.dino_model, config.use_text_embedding,
                config.use_gripper_token, config.use_source_token, config.use_fourier_pe,
                config.fourier_num_frequencies, config.fourier_include_input,
                config.num_transformer_layers, config.dropout, self.device
            )
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")

        self.rng = np.random.default_rng()
        self.GRIPPER_IDX = {
            "aloha": torch.tensor([6, 197, 174]),
            "human": torch.tensor([343, 763, 60]),
            "libero_franka": torch.tensor([1, 2, 0]),  # top, left, right -> left, right, top in agentview
        }

        # For rerun visualization
        self.last_pcd_xyz = None
        self.last_pcd_rgb = None
        self.last_gripper_pcd = None
        self.last_goal_prediction = None
        self.last_goal_gripper_mesh = None

    def _compute_goal_gripper_mesh(self, goal_prediction, gripper_token, joint_state, robot_type):
        """
        Transform gripper mesh from current pose to goal pose for visualization.

        Args:
            goal_prediction: (N, 3) array of predicted goal points
            gripper_token: (10,) array [3 position, 6 rotation (6d), 1 gripper width]
            joint_state: Current joint state for rendering the mesh
            robot_type: Type of robot

        Returns:
            trimesh object transformed to goal pose
        """
        if robot_type != "aloha":
            raise NotImplementedError(f"Goal gripper mesh visualization not implemented for robot_type={robot_type}")
        # Get gripper mesh at current position in world frame
        gripper_mesh = render_aloha_gripper_mesh(np.eye(4), joint_state)

        # Build current gripper pose matrix
        current_pose_mat = np.eye(4)
        current_pose_mat[:3, :3] = rotation_6d_to_matrix(torch.from_numpy(gripper_token[3:9])).numpy()
        current_pose_mat[:3, 3] = gripper_token[:3]

        # Build goal gripper pose matrix from predicted points
        goal_pose = _gripper_pcd_to_token(goal_prediction)
        goal_pose_mat = np.eye(4)
        goal_pose_mat[:3, :3] = rotation_6d_to_matrix(torch.from_numpy(goal_pose[3:9])).numpy()
        goal_pose_mat[:3, 3] = goal_pose[:3]

        # Transform mesh from current pose to goal pose
        gripper_mesh.apply_transform(np.linalg.inv(current_pose_mat))
        gripper_mesh.apply_transform(goal_pose_mat)

        return gripper_mesh

    def _get_text_embedding(self, text, rgb, robot_type, robot_kwargs):
        """Get text embedding with optional Gemini preprocessing and caching"""
        infer_text = self.get_goal_text(text, rgb, robot_type, robot_kwargs)
        if infer_text not in self.text_embedding_cache:
            self.text_embedding_cache[infer_text] = get_siglip_text_embedding(infer_text)
        return self.text_embedding_cache[infer_text]

    def predict(self, text, camera_obs, robot_type, robot_kwargs):
        """
        Args:
            text: Task description
            camera_obs: Dict mapping camera names to {"rgb": ..., "depth": ...}
            robot_type: Robot type (e.g., "aloha")
            robot_kwargs: Additional robot-specific data

        Returns:
            Goal prediction (model-dependent format)
        """
        # Validate that all required cameras are present
        for cam_name in self.camera_names:
            if cam_name not in camera_obs:
                raise ValueError(
                    f"Required camera '{cam_name}' not found in observations. "
                    f"Expected cameras: {self.camera_names}, got: {list(camera_obs.keys())}"
                )

        if self.config.model_type == "articubot":
            return self._predict_articubot(text, camera_obs, robot_type, robot_kwargs)
        elif self.config.model_type == "dino_heatmap":
            raise NotImplementedError("Need to update after changing interface for multiview.")
            return self._predict_dino_heatmap(text, rgb, depth, robot_type, robot_kwargs)
        elif self.config.model_type == "dino_3dgp":
            return self._predict_dino_3dgp(text, camera_obs, robot_type, robot_kwargs)
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def _predict_articubot(self, text, camera_obs, robot_type, robot_kwargs):
        """Prediction using Articubot point cloud model"""
        # Prepare lists for all camera data
        all_pcd_xyz = []
        all_pcd_rgb = []

        # Process each camera
        for cam_idx, cam_name in enumerate(self.camera_names):
            rgb = camera_obs[cam_name]["rgb"]
            depth = camera_obs[cam_name]["depth"]

            # Scale intrinsics if needed
            if self.scaled_Ks[cam_idx] is None:
                self.scaled_Ks[cam_idx] = get_scaled_intrinsics(
                    self.original_Ks[cam_idx], (rgb.shape[0], rgb.shape[1]), TARGET_SHAPE
                )

            # Compute point cloud for this camera in camera frame
            pcd = compute_pcd(rgb, depth, self.scaled_Ks[cam_idx], rgb_preprocess,
                                depth_preprocess, self.device, self.rng,
                                self.config.num_points // self.num_cameras, self.config.max_depth)
            pcd_xyz, pcd_rgb = pcd[:, :3], pcd[:, 3:]

            # Transform xyz to world frame
            pcd_xyz_hom = np.concatenate([pcd_xyz, np.ones((pcd_xyz.shape[0], 1))], axis=1)  # (N, 4)
            pcd_xyz_world = (self.cam_to_worlds[cam_idx] @ pcd_xyz_hom.T).T[:, :3]  # (N, 3)

            all_pcd_xyz.append(pcd_xyz_world)
            all_pcd_rgb.append(pcd_rgb)

        # Concatenate all cameras
        pcd_xyz = np.concatenate(all_pcd_xyz, axis=0)
        pcd_rgb = np.concatenate(all_pcd_rgb, axis=0)

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

        # Use first camera's RGB for text embedding
        text_embed = self._get_text_embedding(text, camera_obs[self.camera_names[0]]["rgb"], robot_type, robot_kwargs)

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

    def _predict_dino_3dgp(self, text, camera_obs, robot_type, robot_kwargs):
        """Prediction using DINO 3D Goal Prediction model"""

        # Prepare lists for all camera data
        all_rgbs = []
        all_depths = []
        all_intrinsics = []
        all_extrinsics = []

        # Process each camera
        for cam_idx, cam_name in enumerate(self.camera_names):
            rgb = camera_obs[cam_name]["rgb"]
            depth = camera_obs[cam_name]["depth"]

            # Scale intrinsics if needed
            if self.scaled_Ks[cam_idx] is None:
                self.scaled_Ks[cam_idx] = get_scaled_intrinsics(
                    self.original_Ks[cam_idx], (rgb.shape[0], rgb.shape[1]), TARGET_SHAPE
                )

            all_rgbs.append(rgb)
            all_depths.append(depth)
            all_intrinsics.append(self.scaled_Ks[cam_idx])
            all_extrinsics.append(self.cam_to_worlds[cam_idx])

        # For visualization, compute pcd from all cameras and transform to world frame
        all_pcd_xyz = []
        all_pcd_rgb = []
        for cam_idx in range(len(all_rgbs)):
            pcd = compute_pcd(all_rgbs[cam_idx], all_depths[cam_idx], self.scaled_Ks[cam_idx],
                             rgb_preprocess, depth_preprocess, self.device, self.rng,
                             1000, self.config.max_depth)
            pcd_xyz, pcd_rgb = pcd[:, :3], pcd[:, 3:]

            # Transform xyz to world frame
            pcd_xyz_hom = np.concatenate([pcd_xyz, np.ones((pcd_xyz.shape[0], 1))], axis=1)  # (N, 4)
            pcd_xyz_world = (self.cam_to_worlds[cam_idx] @ pcd_xyz_hom.T).T[:, :3]  # (N, 3)

            all_pcd_xyz.append(pcd_xyz_world)
            all_pcd_rgb.append(pcd_rgb)

        # Concatenate all cameras
        self.last_pcd_xyz = np.concatenate(all_pcd_xyz, axis=0)
        self.last_pcd_rgb = np.concatenate(all_pcd_rgb, axis=0)

        # Get gripper point cloud if needed
        gripper_token = None
        if self.config.use_gripper_token:
            gripper_pcd = self._get_gripper_pcd(robot_type, robot_kwargs)
            self.last_gripper_pcd = gripper_pcd
            gripper_pcd_ = gripper_pcd[self.GRIPPER_IDX[robot_type]]
            gripper_token = _gripper_pcd_to_token(gripper_pcd_)

        # Get text embedding if needed
        text_embed = None
        if self.config.use_text_embedding:
            text_embed = self._get_text_embedding(text, all_rgbs[0], robot_type, robot_kwargs)

        goal_prediction = inference_dino_3dgp(
            self.model, all_rgbs, all_depths, all_intrinsics, all_extrinsics,
            gripper_token, text, robot_type, self.config.is_gmm,
            self.config.max_depth, self.device
        )

        # Store for rerun visualization
        self.last_goal_prediction = goal_prediction

        # Compute goal gripper mesh for visualization
        if self.config.use_gripper_token:
            joint_state = robot_kwargs["observation.state"]
            self.last_goal_gripper_mesh = self._compute_goal_gripper_mesh(
                goal_prediction, gripper_token, joint_state, robot_type
            )

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

    def project(self, goal_prediction, camera_obs, goal_repr="heatmap"):
        """Project goal prediction to image space"""
        if self.config.model_type == "articubot":
            # Project to each camera
            goal_projections = {}
            for idx, cam_name in enumerate(self.camera_names):
                rgb_shape = camera_obs[cam_name]["rgb"].shape
                goal_projections[cam_name] = self._project_to_camera(goal_prediction, rgb_shape, idx, goal_repr)
            return goal_projections
        elif self.config.model_type == "dino_3dgp":
            # Project to each camera
            goal_projections = {}
            for idx, cam_name in enumerate(self.camera_names):
                rgb_shape = camera_obs[cam_name]["rgb"].shape
                goal_projections[cam_name] = self._project_to_camera(goal_prediction, rgb_shape, idx, goal_repr)
            return goal_projections
        elif self.config.model_type == "dino_heatmap":
            raise NotImplementedError("Not updated after multiview changes.")
            return self._compute_dino_heatmap(goal_prediction, img_shape)
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def _project_to_camera(self, goal_prediction, img_shape, cam_idx, goal_repr="heatmap"):
        """
        Project 3D points in world frame to specific camera's 2D image space.

        Args:
            goal_prediction: (N, 3) array of 3D points in WORLD frame
            img_shape: (H, W, ...) shape of target image
            cam_idx: Index of camera to project to
            goal_repr: "mask" or "heatmap"

        Returns:
            (H, W, 3) projection image
        """
        assert goal_repr in ["mask", "heatmap"]

        # Transform from world frame to this camera's frame
        world_to_cam = np.linalg.inv(self.cam_to_worlds[cam_idx])

        # Convert to homogeneous coordinates
        goal_prediction_hom = np.concatenate([goal_prediction, np.ones((goal_prediction.shape[0], 1))], axis=1)  # (N, 4)

        # Transform to camera frame
        goal_prediction_cam = (world_to_cam @ goal_prediction_hom.T).T  # (N, 4)
        goal_prediction_cam = goal_prediction_cam[:, :3]  # (N, 3)

        # Project to image using this camera's intrinsics
        K = self.original_Ks[cam_idx]
        urdf_proj_hom = (K @ goal_prediction_cam.T).T
        urdf_proj = (urdf_proj_hom / urdf_proj_hom[:, 2:])[:, :2]
        urdf_proj = np.clip(urdf_proj, [0, 0], [img_shape[1] - 1, img_shape[0] - 1]).astype(int)

        goal_gripper_proj = np.zeros((img_shape[0], img_shape[1], 3))
        if goal_repr == "mask":
            goal_gripper_proj[urdf_proj[:, 1], urdf_proj[:, 0]] = 255
        elif goal_repr == "heatmap":
            goal_gripper_proj = generate_heatmap_from_points(urdf_proj, img_shape)
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

    def predict_and_project(self, text, camera_obs, robot_type, robot_kwargs):
        """
        Predict goal and project to all camera image spaces.

        Args:
            text: Task description
            camera_obs: Dict mapping camera names to {"rgb": ..., "depth": ...}
            robot_type: Robot type
            robot_kwargs: Additional robot data

        Returns:
            Dict mapping camera names to goal projections: {cam_name: (H, W, 3) array}
        """
        # Get 3D prediction in world frame
        goal_prediction = self.predict(text, camera_obs, robot_type, robot_kwargs)
        goal_projections = self.project(goal_prediction, camera_obs)

        return goal_projections  # dict[str, np.ndarray]


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

def initialize_dino_heatmap_model(entity, project, checkpoint_type, run_id, dino_model, use_gripper_pcd, use_text_embedding, device):
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
    checkpoint_reference = f"{entity}/{project}/best_{checkpoint_type}_model-{run_id}:best"
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

def initialize_dino_3dgp_model(entity, project, checkpoint_type,
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
            self.image_token_dropout = False # We only do inference here.

    model_cfg = ModelConfig(
        dino_model, use_text_embedding, use_gripper_token,
        use_source_token, use_fourier_pe, fourier_num_frequencies,
        fourier_include_input, num_transformer_layers, dropout
    )
    model = Dino3DGPNetwork(model_cfg)

    artifact_dir = "wandb"
    checkpoint_reference = f"{entity}/{project}/best_{checkpoint_type}_model-{run_id}:best"
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

def monkey_patch_mimicplay(network):
    """
    Monkey-patch in alternate functionality to train Mimicplay baseline.
    """

    def mimicplay_forward(
        self,
        image,
        depth,
        intrinsics,
        extrinsics,
        gripper_token=None,
        text=None,
        source=None,
    ):
        """
        Modified version of forward() for Dino3DGP
        """
        B, N, C, H, W = image.shape

        # Extract DINOv3 features for each camera
        all_patch_features = []
        for cam_idx in range(N):
            with torch.no_grad():
                cam_image = image[:, cam_idx, :, :, :]  # (B, 3, H, W)
                inputs = self.backbone_processor(images=cam_image, return_tensors="pt")
                inputs = {k: v.to(self.backbone.device) for k, v in inputs.items()}
                dino_outputs = self.backbone(**inputs)

            # Get patch features (skip CLS and register tokens)
            patch_features = dino_outputs.last_hidden_state[
                :, 5:
            ]  # (B, 196, dino_hidden_dim)
            all_patch_features.append(patch_features)

        # Concatenate features from all cameras: (B, N*196, dino_hidden_dim)
        patch_features = torch.cat(all_patch_features, dim=1)

        # Get 3D positional encoding for patches (in world frame)
        patch_coords = self.get_patch_centers(
            H, W, intrinsics, depth, extrinsics
        )  # (B, N*196, 3)
        pos_encoding = self.pos_encoder(patch_coords)  # (B, N*196, 128)

        # Combine patch features with positional encoding
        tokens = torch.cat(
            [patch_features, pos_encoding], dim=-1
        )  # (B, N*196, hidden_dim)

        # Apply image token dropout (training only)
        tokens, patch_coords = self.apply_image_token_dropout(tokens, patch_coords, N)

        # Number of tokens T <= N*196
        num_patch_tokens = tokens.shape[1]
        mask = torch.zeros(B, num_patch_tokens, dtype=torch.bool, device=tokens.device)

        # Add language tokens
        if self.use_text_embedding:
            text_tokens = self.text_tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )
            text_tokens = {
                k: v.to(self.text_encoder.device) for k, v in text_tokens.items()
            }
            text_embedding = self.text_encoder(**text_tokens).last_hidden_state

            lang_tokens = self.text_proj(text_embedding)  # (B, J, hidden_dim)
            tokens = torch.cat([tokens, lang_tokens], dim=1)  # (B, T+J, hidden_dim)
            mask = torch.cat([mask, text_tokens["attention_mask"] == 0], dim=1)

        # Add gripper token
        if self.use_gripper_token:
            grip_token = self.gripper_encoder(gripper_token).unsqueeze(
                1
            )  # (B, 1, hidden_dim)
            tokens = torch.cat([tokens, grip_token], dim=1)  # (B, T+J+1, hidden_dim)
            mask = torch.cat(
                [mask, torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)], dim=1
            )

        # Add source token
        if self.use_source_token:
            source_indices = torch.tensor(
                [self.source_to_idx[s] for s in source], device=tokens.device
            )
            source_token = self.source_embeddings(source_indices).unsqueeze(
                1
            )  # (B, 1, hidden_dim)
            tokens = torch.cat([tokens, source_token], dim=1)  # (B, T+J+2, hidden_dim)
            mask = torch.cat(
                [mask, torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)], dim=1
            )

        tokens = torch.cat([tokens, self.registers.expand(B, -1, -1)], dim=1)
        mask = torch.cat(
            [
                mask,
                torch.zeros(
                    B, self.num_registers, dtype=torch.bool, device=tokens.device
                ),
            ],
            dim=1,
        )

        # NEW ###
        # Add CLS token to hold latent plan
        tokens = torch.cat([tokens, self.cls_token.expand(B, -1, -1)], dim=1)
        mask = torch.cat(
            [
                mask,
                torch.zeros(B, 1, dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )

        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens, src_key_padding_mask=mask)

        # NEW ###
        # Take only the CLS token
        latent_plan = tokens[:, -1, :]  # (B, hidden_dim)
        # Predict GMM parameters
        outputs = self.gmm_decoder(latent_plan)

        means = outputs[:, :150].reshape(-1, 5, 30)  # (B, 5, 30)
        raw_scales = outputs[:, 150:300].reshape(-1, 5, 30)  # (B, 5, 30)
        logits = outputs[:, 300:305].reshape(-1, 5)  # (B, 5)

        scales = F.softplus(raw_scales) + 0.0001
        component_dist = D.Independent(D.Normal(means, scales), 1)
        mixture_dist = D.Categorical(logits=logits)
        gmm_dist = D.MixtureSameFamily(mixture_dist, component_dist)

        return latent_plan, gmm_dist

    # Add extra params for latent plan and GMM decoder
    network.cls_token = nn.Parameter(torch.randn(1, 1, network.hidden_dim) * 0.02)
    network.num_modes = 5
    network.pred_timesteps = 10
    # Predict 10-step trajectory of 4th gripper point (xyz only)
    # Output: means (5, 30) + scales (5, 30) + logits (5) = 305 total
    network.gmm_decoder = nn.Sequential(
        nn.Linear(network.hidden_dim, 400),
        nn.Softplus(),
        nn.Linear(400, 400),
        nn.Softplus(),
        nn.Linear(
            400,
            network.num_modes * (network.pred_timesteps * 3 * 2) + network.num_modes,
        ),
        # 5 modes × (10 timesteps × 3 coords × 2 [mean+scale]) + 5 logits = 305
    )
    del network.output_head

    # Add in a separate forward pass for MimicPlay
    network.mimicplay_forward = types.MethodType(mimicplay_forward, network)

    return network

def initialize_mimicplay_model(entity, project, checkpoint_type,
    run_id, dino_model, use_text_embedding, use_gripper_token, use_source_token,
    use_fourier_pe, fourier_num_frequencies, fourier_include_input,
    num_transformer_layers, dropout, device
):
    """Initialize Mimicplay model from wandb artifact"""

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
            self.image_token_dropout = False # We only do inference here.

    model_cfg = ModelConfig(
        dino_model, use_text_embedding, use_gripper_token,
        use_source_token, use_fourier_pe, fourier_num_frequencies,
        fourier_include_input, num_transformer_layers, dropout
    )
    model = monkey_patch_mimicplay(Dino3DGPNetwork(model_cfg))

    artifact_dir = "wandb"
    checkpoint_reference = f"{entity}/{project}/best_{checkpoint_type}_model-{run_id}:best"
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

def inference_dino_3dgp(model, rgbs, depths, intrinsics_list, extrinsics_list,
                        gripper_token, text, robot_type, is_gmm, max_depth, device):
    """
    Run DINO 3D Goal Prediction model inference on RGB+depth and predict 3D goal points.

    Args:
        model (Dino3DGPNetwork): Trained model.
        rgbs (list): List of RGB images [(H, W, 3), ...], uint8, one per camera.
        depths (list): List of depth images [(H, W), ...], uint16 in mm, one per camera.
        intrinsics_list (list): List of camera intrinsics [(3, 3), ...], scaled to 224x224.
        extrinsics_list (list): List of camera extrinsics [(4, 4), ...], T_world_from_camera.
        gripper_token (np.ndarray): Optional gripper token (10,).
        text (str): Optional text caption
        robot_type (str): Robot type (e.g., "aloha", "robot").
        is_gmm (bool): Whether to use GMM sampling or weighted average.
        max_depth (float): Maximum depth threshold in meters.
        device (torch.device): Device for inference.

    Returns:
        pred_points: (4, 3) predicted 3D goal points
    """
    N = len(rgbs)  # Number of cameras

    # Preprocess all RGBs
    rgbs_processed = []
    for rgb in rgbs:
        rgb_ = np.asarray(rgb_preprocess(Image.fromarray(rgb))).copy()
        rgb_ = torch.from_numpy(rgb_).permute(2, 0, 1).float()  # (3, 224, 224)
        rgbs_processed.append(rgb_)

    # Stack into (1, N, 3, 224, 224)
    rgbs_tensor = torch.stack(rgbs_processed, dim=0).unsqueeze(0).to(device)

    # Preprocess all depths
    depths_processed = []
    for depth in depths:
        depth_ = (depth / 1000.0).squeeze().astype(np.float32)  # Convert mm to meters
        depth_ = PIL.Image.fromarray(depth_)
        depth_ = np.asarray(depth_preprocess(depth_)).copy()
        depth_[depth_ > max_depth] = 0  # Mask out far depths
        depth_ = torch.from_numpy(depth_).float()  # (224, 224)
        depths_processed.append(depth_)

    # Stack into (1, N, 224, 224)
    depths_tensor = torch.stack(depths_processed, dim=0).unsqueeze(0).to(device)

    # Stack intrinsics into (1, N, 3, 3)
    intrinsics_tensor = torch.stack([
        torch.from_numpy(K.astype(np.float32)) for K in intrinsics_list
    ], dim=0).unsqueeze(0).to(device)

    # Stack extrinsics into (1, N, 4, 4)
    extrinsics_tensor = torch.stack([
        torch.from_numpy(T.astype(np.float32)) for T in extrinsics_list
    ], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        # Convert gripper_token to torch if provided
        gripper_tok = None
        if gripper_token is not None:
            gripper_tok = torch.from_numpy(gripper_token.astype(np.float32)).unsqueeze(0).to(device)  # (1, 10)

        # Determine source
        source = [robot_type] if model.use_source_token else None

        # Forward through network
        outputs, patch_coords, tokens = model(
            image=rgbs_tensor,
            depth=depths_tensor,
            intrinsics=intrinsics_tensor,
            extrinsics=extrinsics_tensor,
            gripper_token=gripper_tok,
            text=text,
            source=source
        )  # outputs: (1, N*256, 13), patch_coords: (1, N*256, 3) in WORLD frame

        # Sample or get weighted prediction
        if is_gmm:
            # Visualize GMM predictions
            viz_gmm = False
            if viz_gmm:
                visualize_gmm_predictions(patch_coords.permute(0,2,1), outputs)
            pred_points = sample_from_gmm_3dgp(outputs, patch_coords)  # (1, 4, 3)
        else:
            pred_points = get_weighted_prediction_3dgp(outputs, patch_coords)  # (1, 4, 3)

        return pred_points.squeeze(0).cpu().numpy()  # (4, 3) in WORLD frame

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
    import matplotlib.cm as cm

    # Extract point cloud xyz and convert to numpy
    pcd_xyz = pcd[0, :3, :].permute(1, 0).cpu().numpy()  # [N, 3]
    pcd_size = pcd_xyz.shape[0]

    # Extract outputs: 4 displacement vectors (12 channels) + 1 weight (1 channel)
    outputs_np = outputs[0].cpu().numpy()  # [N, 13]
    displacements = outputs_np[:, :12].reshape(-1, 4, 3)  # [N, 4, 3]
    weights = outputs_np[:, 12]  # [N]

    # Softmax the weights over all points
    weights_exp = np.exp(weights - np.max(weights))
    weights = weights_exp / weights_exp.sum()  # [N]

    # Get top 10 weights
    top_k = 10
    top_indices = np.argsort(weights)[-top_k:]  # Get indices of top 10 weights

    # Filter to only top 10
    pcd_xyz_top = pcd_xyz[top_indices]
    displacements_top = displacements[top_indices]
    weights_top = weights[top_indices]

    # Show only one displacement vector per point (first one for simplicity)
    vec_idx = 0
    target_points = pcd_xyz_top + displacements_top[:, vec_idx, :]  # [top_k, 3]

    # Create line segments from each point to its target
    positions = np.zeros((top_k, 2, 3))
    positions[:, 0, :] = pcd_xyz_top  # Start points
    positions[:, 1, :] = target_points  # End points

    # Use matplotlib colormap for weights (viridis is a nice perceptually uniform colormap)
    colormap = cm.get_cmap('viridis')
    colors = (colormap(weights_top)[:, :3] * 255).astype(np.uint8)  # [top_k, 3] in range [0, 255]

    # Log line strips
    rr.log(
        "gmm/displacement_vectors",
        rr.LineStrips3D(
            positions,  # [top_k, 2, 3]
            colors=colors,
            radii=0.002
        )
    )
