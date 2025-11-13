"""
DINOv2-based 3D Goal Prediction Network for goal-conditioned policies.

Copied from lfd3d/models/dino_3dgp.py for inference-only use.
This model predicts 3D goal points using:
- DINOv2 image features
- 3D positional encoding from depth
- Transformer-based fusion
- GMM-based output (256 components, each predicting 4 3D points + weight)
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoImageProcessor, AutoModel


class FourierPositionalEncoding(nn.Module):
    """Fourier feature positional encoding for 3D coordinates"""

    def __init__(self, input_dim=3, num_frequencies=64, include_input=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Frequency bands (geometric progression)
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)

        # Output dimension: input_dim * num_frequencies * 2 (sin + cos) + input_dim (if include_input)
        self.output_dim = input_dim * num_frequencies * 2
        if include_input:
            self.output_dim += input_dim

    def forward(self, coords):
        """
        Args:
            coords: (..., input_dim) 3D coordinates
        Returns:
            encoded: (..., output_dim) Fourier encoded features
        """
        # coords: (..., 3)
        # freq_bands: (num_frequencies,)

        # Compute angular frequencies
        # (..., 3, 1) * (num_frequencies,) -> (..., 3, num_frequencies)
        scaled = coords.unsqueeze(-1) * self.freq_bands

        # Apply sin and cos
        sin_features = torch.sin(2 * np.pi * scaled)  # (..., 3, num_frequencies)
        cos_features = torch.cos(2 * np.pi * scaled)  # (..., 3, num_frequencies)

        # Interleave and flatten
        fourier_features = torch.cat(
            [sin_features, cos_features], dim=-1
        )  # (..., 3, 2*num_frequencies)
        fourier_features = fourier_features.reshape(
            *coords.shape[:-1], -1
        )  # (..., 3*2*num_frequencies)

        if self.include_input:
            fourier_features = torch.cat([coords, fourier_features], dim=-1)

        return fourier_features


class Dino3DGPNetwork(nn.Module):
    """
    DINOv2 + 3D positional encoding + Transformer for 3D goal prediction
    Architecture:
    - Image tokens: DINOv2 patches with 3D PE (x,y,z from depth)
    - Language token: SigLIP embedding (optional)
    - Gripper token: 6DoF pose + gripper width (optional)
    - Source token: learnable embedding for human/robot (optional)
    - Transformer: self-attention blocks
    - Output: N*256 GMM components, each predicting 13-dim (4×3 coords + 1 weight)
    """

    def __init__(self, model_cfg):
        super(Dino3DGPNetwork, self).__init__()

        # DINOv2 backbone
        self.backbone_processor = AutoImageProcessor.from_pretrained(
            model_cfg.dino_model
        )
        self.backbone = AutoModel.from_pretrained(model_cfg.dino_model)
        self.backbone.requires_grad_(False)  # Freeze backbone

        # Get backbone dimensions
        self.pos_encoding_dim = 128
        self.hidden_dim = self.backbone.config.hidden_size + self.pos_encoding_dim
        self.patch_size = self.backbone.config.patch_size

        # Training augmentations
        self.image_token_dropout = model_cfg.image_token_dropout

        # 3D Positional encoding
        if model_cfg.use_fourier_pe:
            # Fourier positional encoding
            fourier_encoder = FourierPositionalEncoding(
                input_dim=3,
                num_frequencies=model_cfg.fourier_num_frequencies,
                include_input=model_cfg.fourier_include_input,
            )
            fourier_dim = fourier_encoder.output_dim
            # Fourier encoder + MLP projection
            self.pos_encoder = nn.Sequential(
                fourier_encoder,
                nn.Linear(fourier_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.pos_encoding_dim),
            )
        else:
            # 3D Positional encoding MLP
            # Input: (x, y, z) coordinates, output: hidden_dim
            self.pos_encoder = nn.Sequential(
                nn.Linear(3, 128),
                nn.ReLU(),
                nn.Linear(128, self.pos_encoding_dim),
            )

        # Language token encoder
        self.use_text_embedding = model_cfg.use_text_embedding
        if self.use_text_embedding:
            self.text_encoder = nn.Sequential(
                nn.Linear(1152, 256),  # SIGLIP input dim
                nn.ReLU(),
                nn.Linear(256, self.hidden_dim),
            )

        # Gripper token encoder (6DoF pose + gripper width = xyz + 6D rot + width = 10dims)
        self.use_gripper_token = model_cfg.use_gripper_token
        if self.use_gripper_token:
            self.gripper_encoder = nn.Sequential(
                nn.Linear(10, 128),
                nn.ReLU(),
                nn.Linear(128, self.hidden_dim),
            )

        # Source token (learnable embeddings for human/robot)
        self.use_source_token = model_cfg.use_source_token
        if self.use_source_token:
            # Learnable embeddings: 0 = human, 1 = robot
            self.source_to_idx = {"human": 0, "aloha": 1, "libero_franka": 2}
            self.source_embeddings = nn.Embedding(3, self.hidden_dim)

        # Transformer blocks (self-attention only)
        self.num_layers = model_cfg.num_transformer_layers
        self.transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=8,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=model_cfg.dropout,
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Output head: predicts 13 dims per component (12 for 4×3 coords + 1 weight)
        self.output_head = nn.Linear(self.hidden_dim, 13)

    def apply_image_token_dropout(self, tokens, patch_coords, num_cameras):
        """
        Apply image token dropout during training.

        Args:
            tokens: (B, N*256, hidden_dim) image tokens
            patch_coords: (B, N*256, 3) patch coordinates
            num_cameras: N - number of cameras

        Returns:
            tokens: (B, T, hidden_dim) tokens after dropout
            patch_coords: (B, T, 3) patch coords after dropout
        """
        if not self.training or not self.image_token_dropout:
            return tokens, patch_coords

        B, total_tokens, hidden_dim = tokens.shape
        tokens_per_camera = 256
        device = tokens.device

        # Sample dropout strategy: 0.6 = no dropout, 0.3 = token dropout, 0.1 = camera dropout
        dropout_type = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]

        if dropout_type == 0:
            # No dropout
            return tokens, patch_coords
        elif dropout_type == 1:
            # Drop 0-30% of all tokens randomly
            dropout_ratio = random.uniform(0.0, 0.3)
            num_tokens_to_keep = int(total_tokens * (1 - dropout_ratio))

            indices = torch.stack(
                [
                    torch.randperm(total_tokens, device=device)[:num_tokens_to_keep]
                    for _ in range(B)
                ]
            )
            batch_idx = torch.arange(B, device=device)[:, None]

            tokens = tokens[batch_idx, indices]
            patch_coords = patch_coords[batch_idx, indices]

            return tokens, patch_coords
        else:
            # Drop one entire camera (only if more than one camera)
            if num_cameras > 1:
                # Randomly select a camera to drop
                camera_to_drop = random.randint(0, num_cameras - 1)
                start_idx = camera_to_drop * tokens_per_camera
                end_idx = start_idx + tokens_per_camera

                # Create mask to keep all tokens except from dropped camera
                mask = torch.ones(total_tokens, dtype=torch.bool, device=device)
                mask[start_idx:end_idx] = False

                # Apply mask
                tokens = tokens[:, mask, :]
                patch_coords = patch_coords[:, mask, :]
            return tokens, patch_coords

    def transform_to_world(self, points_cam, T_world_from_cam):
        """Transform points from camera frame to world frame.

        Args:
            points_cam: (B, N, 3) - points in camera frame
            T_world_from_cam: (B, 4, 4) - transformation matrix

        Returns:
            points_world: (B, N, 3) - points in world frame
        """
        B, N, _ = points_cam.shape
        # Convert to homogeneous coordinates
        ones = torch.ones(B, N, 1, device=points_cam.device)
        points_hom = torch.cat([points_cam, ones], dim=-1)  # (B, N, 4)

        # Apply transformation: (B, 4, 4) @ (B, N, 4) -> (B, N, 4)
        points_world_hom = torch.einsum("bij,bnj->bni", T_world_from_cam, points_hom)

        # Convert back to 3D
        points_world = points_world_hom[:, :, :3]  # (B, N, 3)

        return points_world

    def get_patch_centers(self, H, W, intrinsics, depth, extrinsics):
        """
        Compute 3D coordinates for patch centers using depth (multi-camera support).

        Args:
            H, W: image height and width
            intrinsics: (B, N, 3, 3) camera intrinsics for N cameras
            depth: (B, N, H, W) depth maps for N cameras
            extrinsics: (B, N, 4, 4) camera-to-world transforms

        Returns:
            patch_coords: (B, N*num_patches, 3) 3D coordinates in WORLD frame
        """
        B, N, _, _ = depth.shape
        device = depth.device

        # Calculate patch grid size (DINOv2 uses 16×16 patches for 224×224 image)
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_patches = h_patches * w_patches  # 256 for 224x224 with patch_size=14

        # Get center pixel of each patch
        y_centers = (
            torch.arange(h_patches, device=device) * self.patch_size
            + self.patch_size // 2
        )
        x_centers = (
            torch.arange(w_patches, device=device) * self.patch_size
            + self.patch_size // 2
        )
        yy, xx = torch.meshgrid(y_centers, x_centers, indexing="ij")

        # Flatten to (num_patches, 2)
        pixel_coords = torch.stack(
            [xx.flatten(), yy.flatten()], dim=1
        )  # (num_patches, 2)

        # Process each camera
        all_coords_world = []
        for cam_idx in range(N):
            # Sample depth at patch centers for this camera
            pixel_coords_batch = pixel_coords.unsqueeze(0).expand(
                B, -1, -1
            )  # (B, num_patches, 2)
            y_idx = pixel_coords_batch[:, :, 1].long()
            x_idx = pixel_coords_batch[:, :, 0].long()

            depth_cam = depth[:, cam_idx, :, :]  # (B, H, W)
            depth_values = depth_cam[
                torch.arange(B, device=device).unsqueeze(1), y_idx, x_idx
            ]  # (B, num_patches)

            # Unproject to 3D in camera frame
            K = intrinsics[:, cam_idx, :, :]  # (B, 3, 3)
            fx = K[:, 0, 0].unsqueeze(1)  # (B, 1)
            fy = K[:, 1, 1].unsqueeze(1)
            cx = K[:, 0, 2].unsqueeze(1)
            cy = K[:, 1, 2].unsqueeze(1)

            x_3d = (pixel_coords_batch[:, :, 0] - cx) * depth_values / fx
            y_3d = (pixel_coords_batch[:, :, 1] - cy) * depth_values / fy
            z_3d = depth_values

            patch_coords_cam = torch.stack(
                [x_3d, y_3d, z_3d], dim=2
            ).float()  # (B, num_patches, 3)

            # Transform to world frame
            T_world_from_cam = extrinsics[:, cam_idx, :, :]  # (B, 4, 4)
            patch_coords_world = self.transform_to_world(
                patch_coords_cam, T_world_from_cam
            )

            all_coords_world.append(patch_coords_world)

        # Concatenate all cameras: (B, N*num_patches, 3)
        patch_coords = torch.cat(all_coords_world, dim=1)

        return patch_coords

    def forward(
        self,
        image,
        depth,
        intrinsics,
        extrinsics,
        gripper_token=None,
        text_embedding=None,
        source=None,
    ):
        """
        Multi-camera forward pass.

        Args:
            image: (B, N, 3, H, W) RGB images for N cameras
            depth: (B, N, H, W) depth maps for N cameras
            intrinsics: (B, N, 3, 3) camera intrinsics
            extrinsics: (B, N, 4, 4) camera-to-world transforms
            gripper_token: (B, 10) [6DoF pose (3 pos + 6 rot6d) + gripper width]
            text_embedding: (B, 1152) SigLIP embedding
            source: list of strings ["human" or "aloha"]

        Returns:
            outputs: (B, T, 13) GMM parameters for all cameras
            patch_coords: (B, T, 3) patch center 3D coordinates in WORLD frame
        """
        B, N, C, H, W = image.shape

        # Extract DINOv2 features for each camera
        all_patch_features = []
        for cam_idx in range(N):
            with torch.no_grad():
                cam_image = image[:, cam_idx, :, :, :]  # (B, 3, H, W)
                inputs = self.backbone_processor(images=cam_image, return_tensors="pt")
                inputs = {k: v.to(self.backbone.device) for k, v in inputs.items()}
                dino_outputs = self.backbone(**inputs)

            # Get patch features (skip CLS token)
            patch_features = dino_outputs.last_hidden_state[
                :, 1:
            ]  # (B, 256, dino_hidden_dim)
            all_patch_features.append(patch_features)

        # Concatenate features from all cameras: (B, N*256, dino_hidden_dim)
        patch_features = torch.cat(all_patch_features, dim=1)

        # Get 3D positional encoding for patches (in world frame)
        patch_coords = self.get_patch_centers(
            H, W, intrinsics, depth, extrinsics
        )  # (B, N*256, 3)
        pos_encoding = self.pos_encoder(patch_coords)  # (B, N*256, 128)

        # Combine patch features with positional encoding
        tokens = torch.cat(
            [patch_features, pos_encoding], dim=-1
        )  # (B, N*256, hidden_dim)

        # Apply image token dropout (training only)
        tokens, patch_coords = self.apply_image_token_dropout(tokens, patch_coords, N)

        # Number of tokens T <= N*256
        num_patch_tokens = tokens.shape[1]

        # Add language token
        if self.use_text_embedding:
            lang_token = self.text_encoder(text_embedding).unsqueeze(
                1
            )  # (B, 1, hidden_dim)
            tokens = torch.cat([tokens, lang_token], dim=1)  # (B, T+1, hidden_dim)

        # Add gripper token
        if self.use_gripper_token:
            grip_token = self.gripper_encoder(gripper_token).unsqueeze(
                1
            )  # (B, 1, hidden_dim)
            tokens = torch.cat([tokens, grip_token], dim=1)  # (B, T+2, hidden_dim)

        # Add source token
        if self.use_source_token:
            source_indices = torch.tensor(
                [self.source_to_idx[s] for s in source], device=tokens.device
            )
            source_token = self.source_embeddings(source_indices).unsqueeze(
                1
            )  # (B, 1, hidden_dim)
            tokens = torch.cat([tokens, source_token], dim=1)  # (B, T+3, hidden_dim)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Take only the patch tokens (throw away language, gripper, source tokens)
        tokens = tokens[:, :num_patch_tokens]  # (B, T, hidden_dim)

        # Predict GMM parameters
        outputs = self.output_head(tokens)  # (B, T, 13)

        return outputs, patch_coords


def sample_from_gmm_3dgp(outputs, patch_coords):
    """
    Sample from GMM by selecting a component and using its mean.
    Args:
        outputs: (B, 256, 13) GMM parameters
        patch_coords: (B, 256, 3) patch center coordinates
    Returns:
        pred_points: (B, 4, 3) sampled goal points
    """
    B, num_components, _ = outputs.shape
    device = outputs.device

    # Parse outputs
    pred_displacement = outputs[:, :, :-1].reshape(B, num_components, 4, 3)
    weights = outputs[:, :, -1]

    # Softmax weights
    weights = F.softmax(weights, dim=1)

    # Sample component indices
    sampled_indices = torch.multinomial(weights, num_samples=1)  # (B, 1)
    batch_indices = torch.arange(B, device=device).unsqueeze(1)

    # Get sampled displacement and add to patch center
    sampled_disp = pred_displacement[batch_indices, sampled_indices].squeeze(
        1
    )  # (B, 4, 3)
    sampled_patch = patch_coords[batch_indices, sampled_indices].squeeze(
        1
    )  # (B, 3)

    pred_points = sampled_disp + sampled_patch.unsqueeze(1)  # (B, 4, 3)
    return pred_points


def get_weighted_prediction_3dgp(outputs, patch_coords):
    """
    Get weighted average prediction (non-GMM mode).
    Args:
        outputs: (B, 256, 13) GMM parameters
        patch_coords: (B, 256, 3) patch center coordinates
    Returns:
        pred_points: (B, 4, 3) weighted average goal points
    """
    B, num_components, _ = outputs.shape

    pred_displacement = outputs[:, :, :-1].reshape(B, num_components, 4, 3)
    weights = F.softmax(outputs[:, :, -1], dim=1)

    # Weighted average
    patch_coords_expanded = patch_coords[:, :, None, :]
    pred_abs = pred_displacement + patch_coords_expanded
    pred_points = (weights[:, :, None, None] * pred_abs).sum(dim=1)
    return pred_points
