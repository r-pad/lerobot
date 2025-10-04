"""
DINO-based heatmap prediction network for goal-conditioned policies.

Copied from lfd3d to avoid circular dependency.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoImageProcessor, AutoModel


def sample_from_heatmap(heatmap):
    """
    Sample 2D pixel coordinates from heatmap using multinomial sampling.

    Args:
        heatmap: (B, 1, H, W) predicted heatmap

    Returns:
        coords: (B, 2) sampled pixel coordinates [x, y]
    """
    B, _, H, W = heatmap.shape

    # Flatten spatial dimensions and apply softmax
    logits = heatmap.squeeze(1).flatten(1)  # (B, H*W)
    probs = F.softmax(logits, dim=1)

    # Multinomial sampling
    sampled_indices = torch.multinomial(probs, 1).squeeze(1)  # (B,)

    # Convert flat indices back to 2D coordinates
    y_coords = sampled_indices // W
    x_coords = sampled_indices % W

    # Stack to get (B, 2) [x, y] format
    coords = torch.stack([x_coords, y_coords], dim=1)

    return coords


class SimplePointNet(nn.Module):
    """Simple PointNet encoder for 3D point clouds"""

    def __init__(self, input_dim, output_dim):
        super(SimplePointNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Make translation invariant
        centroid = x.mean(dim=1, keepdim=True)
        x = x - centroid

        # x: (B, N, input_dim) -> (B, input_dim, N)
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        # Global max pooling
        x = torch.max(x, 2)[0]  # (B, output_dim)
        return x


class DinoHeatmapNetwork(nn.Module):
    """
    DINO + DPT-style decoder for dense heatmap prediction.

    Predicts a single-channel heatmap from RGB image input, optionally
    conditioned on gripper point cloud and/or text embedding.
    """

    def __init__(self, model_cfg):
        super(DinoHeatmapNetwork, self).__init__()

        # DINO backbone
        self.backbone_processor = AutoImageProcessor.from_pretrained(
            model_cfg.dino_model
        )
        self.backbone = AutoModel.from_pretrained(model_cfg.dino_model)
        self.backbone.requires_grad_(False)  # Freeze backbone

        # Get backbone dimensions
        hidden_dim = self.backbone.config.hidden_size

        # Point encoder for hand/gripper poses
        self.use_gripper_pcd = model_cfg.use_gripper_pcd
        self.encoded_point_dim = 128
        if self.use_gripper_pcd:
            self.point_encoder = SimplePointNet(3, self.encoded_point_dim)

        # Language conditioning
        self.use_text_embedding = model_cfg.use_text_embedding
        self.encoded_text_dim = 128
        if self.use_text_embedding:
            self.text_encoder = nn.Linear(
                1152, self.encoded_text_dim
            )  # SIGLIP input dim

        # Cross-attention fusion with layer norm for stability
        if self.use_text_embedding:
            self.text_cross_attn = nn.MultiheadAttention(
                hidden_dim, 8, batch_first=True
            )
            self.text_norm_pre = nn.LayerNorm(hidden_dim)
            self.text_norm_post = nn.LayerNorm(hidden_dim)

        if self.use_gripper_pcd:
            self.point_cross_attn = nn.MultiheadAttention(
                hidden_dim, 8, batch_first=True
            )
            self.point_norm_pre = nn.LayerNorm(hidden_dim)
            self.point_norm_post = nn.LayerNorm(hidden_dim)

        # Project conditioning features to hidden_dim
        if self.use_text_embedding:
            self.text_proj = nn.Linear(self.encoded_text_dim, hidden_dim)
        if self.use_gripper_pcd:
            self.point_proj = nn.Linear(self.encoded_point_dim, hidden_dim)

        # DPT-style decoder
        self.decoder = nn.Sequential(
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, 256, 3, 1, 1),
            nn.ReLU(),
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            # 64x64 -> 128x128
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            # 128x128 -> 224x224
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 1, 3, 1, 1),
        )

        # Initialize cross-attention weights for stability
        self._init_cross_attention_weights()

    def _init_cross_attention_weights(self):
        """Initialize cross-attention layers with smaller weights for training stability"""
        if self.use_text_embedding:
            # Scale down attention weights
            nn.init.xavier_uniform_(self.text_cross_attn.in_proj_weight, gain=0.1)
            nn.init.xavier_uniform_(self.text_cross_attn.out_proj.weight, gain=0.1)

        if self.use_gripper_pcd:
            # Scale down attention weights
            nn.init.xavier_uniform_(self.point_cross_attn.in_proj_weight, gain=0.1)
            nn.init.xavier_uniform_(self.point_cross_attn.out_proj.weight, gain=0.1)

    def forward(self, image, gripper_pcd=None, text_embedding=None):
        """
        Forward pass through the network.

        Args:
            image: List of RGB images or batch tensor (B, 3, H, W)
            gripper_pcd: Optional gripper point cloud (B, N, 3)
            text_embedding: Optional text embedding (B, D)

        Returns:
            heatmap: Predicted single-channel heatmap (B, 1, H, W)
        """
        # Extract features from DINO
        with torch.no_grad():
            inputs = self.backbone_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.backbone.device) for k, v in inputs.items()}
            outputs = self.backbone(**inputs)

        features = outputs.last_hidden_state[:, 1:]  # skip CLS token, (B, 256, 768)
        B, L, D = features.shape

        # Cross-attention fusion
        fused = features

        # Text cross-attention
        if self.use_text_embedding and text_embedding is not None:
            text_feat = self.text_encoder(text_embedding)  # (B, 128)
            text_feat = self.text_proj(text_feat)  # (B, hidden_dim)
            text_feat = text_feat.unsqueeze(1)  # (B, 1, hidden_dim)

            # Cross-attention: query=visual_features, key/value=text_features
            normed_fused = self.text_norm_pre(fused)
            attn_out, _ = self.text_cross_attn(normed_fused, text_feat, text_feat)
            fused = self.text_norm_post(fused + attn_out)  # Residual connection

        # Point cross-attention
        if self.use_gripper_pcd and gripper_pcd is not None:
            point_feat = self.point_encoder(gripper_pcd)  # (B, 128)
            point_feat = self.point_proj(point_feat)  # (B, hidden_dim)
            point_feat = point_feat.unsqueeze(1)  # (B, 1, hidden_dim)

            # Cross-attention: query=visual_features, key/value=point_features
            normed_fused = self.point_norm_pre(fused)
            attn_out, _ = self.point_cross_attn(normed_fused, point_feat, point_feat)
            fused = self.point_norm_post(fused + attn_out)  # Residual connection

        # Reshape to spatial format (assuming square patches)
        h = w = int(L**0.5)
        fused = fused.transpose(1, 2).reshape(B, D, h, w)

        # Decode to heatmap
        heatmap = self.decoder(fused)  # (B, 1, H, W)

        return heatmap
