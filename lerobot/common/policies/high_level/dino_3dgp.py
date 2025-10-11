"""
DINOv2-based 3D Goal Prediction Network for goal-conditioned policies.

Copied from lfd3d/models/dino_3dgp.py for inference-only use.
This model predicts 3D goal points using:
- DINOv2 image features
- 3D positional encoding from depth
- Transformer-based fusion
- GMM-based output (256 components, each predicting 4 3D points + weight)
"""

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
    - Output: 256 GMM components, each predicting 13-dim (4×3 coords + 1 weight)
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
        self.num_components = 256  # Fixed number of GMM components

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
            self.source_to_idx = {"human": 0, "aloha": 1}
            self.source_embeddings = nn.Embedding(2, self.hidden_dim)

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

    def get_patch_centers(self, H, W, intrinsics, depth):
        """
        Compute 3D coordinates for patch centers using depth.
        Args:
            H, W: image height and width
            intrinsics: (B, 3, 3) camera intrinsics
            depth: (B, H, W) depth map
        Returns:
            patch_coords: (B, num_patches, 3) 3D coordinates
        """
        B = depth.shape[0]
        device = depth.device

        # Calculate patch grid size (DINOv2 uses 16×16 patches for 224×224 image)
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

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

        # Sample depth at patch centers
        pixel_coords_batch = pixel_coords.unsqueeze(0).expand(
            B, -1, -1
        )  # (B, num_patches, 2)
        y_idx = pixel_coords_batch[:, :, 1].long()
        x_idx = pixel_coords_batch[:, :, 0].long()

        depth_values = depth[
            torch.arange(B, device=device).unsqueeze(1), y_idx, x_idx
        ]  # (B, num_patches)

        # Unproject to 3D
        fx = intrinsics[:, 0, 0].unsqueeze(1)  # (B, 1)
        fy = intrinsics[:, 1, 1].unsqueeze(1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)

        x_3d = (pixel_coords_batch[:, :, 0] - cx) * depth_values / fx
        y_3d = (pixel_coords_batch[:, :, 1] - cy) * depth_values / fy
        z_3d = depth_values

        patch_coords = torch.stack(
            [x_3d, y_3d, z_3d], dim=2
        ).float()  # (B, num_patches, 3)

        return patch_coords

    def forward(
        self,
        image,
        depth,
        intrinsics,
        gripper_token=None,
        text_embedding=None,
        source=None,
    ):
        """
        Args:
            image: (B, 3, H, W) RGB image
            depth: (B, H, W) depth map
            intrinsics: (B, 3, 3) camera intrinsics
            gripper_token: (B, 10) [6DoF pose (3 pos + 6 rot6d) + gripper width]
            text_embedding: (B, 1152) SigLIP embedding
            source: list of strings ["human" or "aloha"]
        Returns:
            outputs: (B, 256, 13) GMM parameters
            patch_coords: (B, 256, 3) patch center 3D coordinates
        """
        B, _, H, W = image.shape

        # Extract DINOv2 features
        with torch.no_grad():
            inputs = self.backbone_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.backbone.device) for k, v in inputs.items()}
            dino_outputs = self.backbone(**inputs)

        # Get patch features (skip CLS token)
        patch_features = dino_outputs.last_hidden_state[
            :, 1:
        ]  # (B, 256, dino_hidden_dim)

        # Get 3D positional encoding for patches
        patch_coords = self.get_patch_centers(H, W, intrinsics, depth)
        pos_encoding = self.pos_encoder(patch_coords)  # (B, 256, 128)

        # Combine patch features with positional encoding
        tokens = torch.cat(
            [patch_features, pos_encoding], dim=-1
        )  # (B, 256, hidden_dim)

        # Add language token
        if self.use_text_embedding:
            lang_token = self.text_encoder(text_embedding).unsqueeze(
                1
            )  # (B, 1, hidden_dim)
            tokens = torch.cat([tokens, lang_token], dim=1)  # (B, 257, hidden_dim)

        # Add gripper token
        if self.use_gripper_token:
            grip_token = self.gripper_encoder(gripper_token).unsqueeze(
                1
            )  # (B, 1, hidden_dim)
            tokens = torch.cat([tokens, grip_token], dim=1)  # (B, 258, hidden_dim)

        # Add source token
        if self.use_source_token:
            source_indices = torch.tensor(
                [self.source_to_idx[s] for s in source], device=tokens.device
            )
            source_token = self.source_embeddings(source_indices).unsqueeze(
                1
            )  # (B, 1, hidden_dim)
            tokens = torch.cat([tokens, source_token], dim=1)  # (B, 259, hidden_dim)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Take only the first 256 tokens (throw away language and gripper tokens)
        tokens = tokens[:, :256]  # (B, 256, hidden_dim)

        # Predict GMM parameters
        outputs = self.output_head(tokens)  # (B, 256, 13)

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
