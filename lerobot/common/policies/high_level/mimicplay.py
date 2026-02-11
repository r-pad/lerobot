"""
MimicPlay: latent-plan conditioned diffusion policy.

Model definition (monkey-patch on Dino3DGPNetwork), initialization from wandb,
batch-preparation helpers for training and inference, and visualization utilities.
"""

import types

import einops
import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
import wandb
from torch import nn

from lerobot.common.policies.high_level.dino_3dgp import Dino3DGPNetwork
from lerobot.common.policies.high_level.high_level_wrapper import (
    TARGET_SHAPE,
    _get_gripper_pcd,
    _gripper_pcd_to_token,
    depth_preprocess,
    get_scaled_intrinsics,
    rgb_preprocess,
)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

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
        # 5 modes x (10 timesteps x 3 coords x 2 [mean+scale]) + 5 logits = 305
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


# ---------------------------------------------------------------------------
# Batch preparation helpers
# ---------------------------------------------------------------------------

def prepare_mimicplay_batch_inference(
    batch, camera_names, intrinsics, extrinsics, scaled_Ks,
    gripper_idx, robot_type, max_depth,
):
    """Prepare tensors for mimicplay_forward from an inference-time batch.

    Args:
        batch: dict from environment rollout.
        camera_names: list of camera name strings.
        intrinsics: list of (3,3) numpy intrinsics per camera (original scale).
        extrinsics: list of (4,4) numpy extrinsics per camera.
        scaled_Ks: mutable list[Tensor | None] — lazily populated with scaled
            intrinsics on first call (same semantics as DiffusionPolicy.scaled_Ks).
        gripper_idx: dict mapping robot_type -> Tensor of 3 indices.
        robot_type: str (e.g. "aloha").
        max_depth: float, depth clipping threshold in meters.

    Returns:
        rgbs:        (1, num_cams, 3, 224, 224)
        depths:      (1, num_cams, 224, 224)
        intrinsics_t: (1, num_cams, 3, 3)
        extrinsics_t: (1, num_cams, 4, 4)
        gripper_token: (1, 10)
        gripper_pcd_selected: (3, 3) numpy — the 3 key gripper points (for viz)
    """
    rgbs, depths_list = [], []
    for cam_idx, cam in enumerate(camera_names):
        rgb_key = f"observation.images.{cam}.color"
        depth_key = f"observation.images.{cam}.transformed_depth"

        rgb = (batch[rgb_key].squeeze() * 255).to(torch.uint8)
        depth = batch[depth_key].to(torch.float32).squeeze(0)
        depth[depth > max_depth] = 0

        rgbs.append(rgb_preprocess(rgb))          # (3, H, W)
        depths_list.append(depth_preprocess(depth))  # (H, W)

        if scaled_Ks[cam_idx] is None:
            img_shape = batch[rgb_key].shape[-2:]
            scaled_Ks[cam_idx] = torch.from_numpy(
                get_scaled_intrinsics(intrinsics[cam_idx], img_shape, TARGET_SHAPE)
            ).to(rgb.device)

    rgbs = torch.stack(rgbs, dim=0)[None]                                   # (1, N, 3, H, W)
    depths_t = torch.stack(depths_list, dim=0)[None]                        # (1, N, H, W)
    intrinsics_t = torch.stack(scaled_Ks).to(rgbs.device).float()[None]     # (1, N, 3, 3)
    extrinsics_t = torch.from_numpy(
        np.stack(extrinsics)
    ).to(rgbs.device).float()[None]                                         # (1, N, 4, 4)

    # Gripper token
    robot_kwargs = {"observation.state": batch["observation.state"].squeeze()}
    gripper_pcd = _get_gripper_pcd(robot_type, robot_kwargs)
    gripper_pcd_selected = gripper_pcd[gripper_idx[robot_type]]
    gripper_token = _gripper_pcd_to_token(gripper_pcd_selected)             # (10,)
    gripper_token = torch.from_numpy(
        gripper_token.astype(np.float32)
    ).to(rgbs.device).unsqueeze(0)                                          # (1, 10)

    return rgbs, depths_t, intrinsics_t, extrinsics_t, gripper_token, gripper_pcd_selected


def prepare_mimicplay_batch_train(batch, obs_key, gripper_idx, max_depth):
    """Prepare tensors for mimicplay_forward from a training batch.

    Args:
        batch: training batch dict with (B, S, ...) shaped tensors.
        obs_key: observation key to read batch_size / n_obs_steps from.
        gripper_idx: dict mapping robot_type -> Tensor of 3 indices.
        max_depth: float, depth clipping threshold in meters.

    Returns:
        rgbs:           (B*S, num_cams, 3, 224, 224)
        depths:         (B*S, num_cams, 224, 224)
        intrinsics_t:   (B*S, num_cams, 3, 3)
        extrinsics_t:   (B*S, num_cams, 4, 4)
        gripper_tokens: (B*S, 10)
        text:           list[str] of length B*S
        source:         list[str] of length B*S
        batch_size:     int
        n_obs_steps:    int
    """
    batch_size = batch[obs_key].shape[0]
    n_obs_steps = batch[obs_key].shape[1]
    gripper_pcd_key = "observation.points.gripper_pcds"

    # Discover non-wrist cameras from batch keys
    camera_names = [
        s.split('.')[2] for s in batch.keys()
        if s.startswith('observation.images.cam_')
        and s.endswith('.color')
        and 'wrist' not in s
    ]

    rgbs, depths_list, intrinsics_list, extrinsics_list = [], [], [], []
    for i, cam in enumerate(camera_names):
        rgb_key = f"observation.images.{cam}.color"
        depth_key = f"observation.images.{cam}.transformed_depth"
        intrinsics_key = f"observation.{cam}.intrinsics"
        extrinsics_key = f"observation.{cam}.extrinsics"

        rgb = (einops.rearrange(batch[rgb_key], "b s c h w -> (b s) c h w") * 255).to(torch.uint8)
        depth = einops.rearrange(batch[depth_key], "b s c h w -> (b s) h w c").to(torch.float32).squeeze()
        depth[depth > max_depth] = 0

        rgbs.append(rgb_preprocess(rgb))          # (B*S, 3, H, W)
        depths_list.append(depth_preprocess(depth))  # (B*S, H, W)

        img_shape = batch[rgb_key].shape[-2:]
        scaled_K = torch.from_numpy(
            get_scaled_intrinsics(
                batch[intrinsics_key][0, 0].cpu().numpy(), img_shape, TARGET_SHAPE
            )
        ).to(rgb.device)
        intrinsics_list.append(scaled_K[None].repeat(batch_size, 1, 1))
        extrinsics_list.append(batch[extrinsics_key][:, -1])

    rgbs = torch.stack(rgbs, dim=1)           # (B*S, num_cams, C, H, W)
    depths_t = torch.stack(depths_list, dim=1)  # (B*S, num_cams, H, W)

    intrinsics_t = torch.stack(intrinsics_list, axis=1)   # (B, num_cams, 3, 3)
    intrinsics_t = einops.repeat(intrinsics_t, "b n h w -> (b s) n h w", s=n_obs_steps)
    extrinsics_t = torch.stack(extrinsics_list, axis=1)   # (B, num_cams, 4, 4)
    extrinsics_t = einops.repeat(extrinsics_t, "b n h w -> (b s) n h w", s=n_obs_steps)

    text = batch['task'] * n_obs_steps
    source = batch['embodiment'] * n_obs_steps

    gripper_pcd = einops.rearrange(batch[gripper_pcd_key], "b s n p -> (b s) n p")
    gripper_tokens = []
    for i in range(batch_size * n_obs_steps):
        gripper_pcd_ = gripper_pcd[i][gripper_idx[source[i]]].cpu().numpy()
        gripper_token = _gripper_pcd_to_token(gripper_pcd_)  # (10,)
        gripper_token = torch.from_numpy(gripper_token.astype(np.float32)).to(rgbs.device)
        gripper_tokens.append(gripper_token)
    gripper_tokens = torch.stack(gripper_tokens, dim=0)  # (B*S, 10)

    return (rgbs, depths_t, intrinsics_t, extrinsics_t,
            gripper_tokens, text, source, batch_size, n_obs_steps)


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------

def _unproject_cameras_to_pointcloud(rgbs, depths, intrinsics, extrinsics, max_depth, sample_idx=0):
    """Shared helper: unproject multi-camera RGBD to a world-frame point cloud.

    Args:
        rgbs:       (B, num_cams, C, H, W) uint8 [0,255]
        depths:     (B, num_cams, H, W) float meters
        intrinsics: (B, num_cams, 3, 3)
        extrinsics: (B, num_cams, 4, 4)  T_world_from_cam
        max_depth:  float
        sample_idx: which batch element to use

    Returns:
        pcd_xyz: (M, 3)  numpy
        pcd_rgb: (M, 3)  numpy uint8
    """
    _, num_cams, C, H, W = rgbs.shape
    all_pcd_xyz, all_pcd_rgb = [], []

    for cam_idx in range(num_cams):
        depth = depths[sample_idx, cam_idx].cpu().numpy()
        K = intrinsics[sample_idx, cam_idx].cpu().numpy()
        T_world_cam = extrinsics[sample_idx, cam_idx].cpu().numpy()
        rgb = rgbs[sample_idx, cam_idx].permute(1, 2, 0).cpu().numpy()

        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        x_flat, y_flat, z_flat = x_coords.flatten(), y_coords.flatten(), depth.flatten()

        valid = (z_flat > 0) & (z_flat < max_depth)
        x_flat, y_flat, z_flat = x_flat[valid], y_flat[valid], z_flat[valid]

        pixels = np.stack([x_flat, y_flat, np.ones_like(x_flat)], axis=0)
        points_cam = (np.linalg.inv(K) @ pixels) * z_flat
        points_cam = points_cam.T

        points_hom = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)
        points_world = (T_world_cam @ points_hom.T).T[:, :3]

        all_pcd_xyz.append(points_world)
        all_pcd_rgb.append(rgb.reshape(-1, 3)[valid])

    pcd_xyz = np.concatenate(all_pcd_xyz, axis=0)
    pcd_rgb = np.concatenate(all_pcd_rgb, axis=0)

    # Downsample for viz
    if pcd_xyz.shape[0] > 5000:
        idx = np.random.choice(pcd_xyz.shape[0], 5000, replace=False)
        pcd_xyz, pcd_rgb = pcd_xyz[idx], pcd_rgb[idx]

    return pcd_xyz, pcd_rgb


def visualize_mimicplay_rerun(rgbs, depths, intrinsics, extrinsics,
                               pred_eef_dist, gripper_pcd, max_depth):
    """Visualize point clouds and predicted EEF trajectory in rerun.

    Args:
        rgbs:       (B, num_cams, C, H, W) uint8 [0,255]
        depths:     (B, num_cams, H, W) float meters
        intrinsics: (B, num_cams, 3, 3)
        extrinsics: (B, num_cams, 4, 4)
        pred_eef_dist: MixtureSameFamily GMM distribution
        gripper_pcd: (3, 3) numpy — selected gripper key-points
        max_depth: float
    """
    import rerun as rr

    pcd_xyz, pcd_rgb = _unproject_cameras_to_pointcloud(
        rgbs, depths, intrinsics, extrinsics, max_depth
    )

    rr.log("mimicplay/scene_pointcloud", rr.Points3D(pcd_xyz, colors=pcd_rgb.astype(np.uint8)))
    rr.log("mimicplay/gripper_pcd", rr.Points3D(gripper_pcd, colors=[0, 255, 0], radii=0.01))

    # Best-mode mean trajectory
    mean_trajectory = pred_eef_dist.component_distribution.mean  # (B, 5, 30)
    weights = pred_eef_dist.mixture_distribution.probs            # (B, 5)
    best_mode = weights[0].argmax().item()
    mean_traj = mean_trajectory[0, best_mode].cpu().numpy().reshape(10, 3)

    rr.log("mimicplay/pred_eef_trajectory", rr.LineStrips3D(
        [mean_traj], colors=[[255, 0, 0]], radii=0.005
    ))
    rr.log("mimicplay/pred_eef_points", rr.Points3D(
        mean_traj, colors=[[0, 0, 255]] * 10, radii=0.008
    ))


def visualize_mimicplay_open3d(rgbs, depths, intrinsics, extrinsics,
                                pred_eef_dist, gripper_pcd, max_depth, sample_idx=0):
    """Visualize point clouds and predicted EEF trajectory using open3d (training).

    Args:
        rgbs:       (B, num_cams, C, H, W) uint8 [0,255]
        depths:     (B, num_cams, H, W) float meters
        intrinsics: (B, num_cams, 3, 3)
        extrinsics: (B, num_cams, 4, 4)
        pred_eef_dist: MixtureSameFamily GMM distribution
        gripper_pcd: (N, 3) tensor — gripper point cloud
        max_depth:  float
        sample_idx: which batch element to visualize
    """
    import open3d as o3d

    pcd_xyz, pcd_rgb = _unproject_cameras_to_pointcloud(
        rgbs, depths, intrinsics, extrinsics, max_depth, sample_idx=sample_idx
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
    pcd.colors = o3d.utility.Vector3dVector(pcd_rgb.astype(np.float64) / 255.0)

    # Best-mode mean trajectory
    mean_trajectory = pred_eef_dist.component_distribution.mean  # (B, 5, 30)
    weights = pred_eef_dist.mixture_distribution.probs            # (B, 5)
    best_mode = weights[sample_idx].argmax().item()
    mean_traj = mean_trajectory[sample_idx, best_mode].cpu().numpy().reshape(10, 3)

    lines = [[i, i + 1] for i in range(9)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(mean_traj)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))

    traj_spheres = []
    for i, pt in enumerate(mean_traj):
        t = i / 9.0
        color = [1 - t, t, 0]
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        traj_spheres.append(sphere)

    gripper_pcd_o3d = o3d.geometry.PointCloud()
    gripper_pcd_o3d.points = o3d.utility.Vector3dVector(gripper_pcd.cpu().numpy())
    gripper_pcd_o3d.paint_uniform_color([0, 1, 0])

    geometries = [pcd, line_set, gripper_pcd_o3d] + traj_spheres
    o3d.visualization.draw_geometries(geometries, window_name="Training pred_eef visualization")
