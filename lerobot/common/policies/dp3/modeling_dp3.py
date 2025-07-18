import torch
import torch.nn as nn
from typing import Dict
from collections import deque

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import populate_queues

# Import DP3 components
from lerobot.common.policies.dp3.model.vision.pointnet_extractor import DP3Encoder
from lerobot.common.policies.dp3.model.diffusion.conditional_unet1d import ConditionalUnet1D

from lerobot.common.policies.dp3.configuration_dp3 import DP3Config
from lerobot.common.constants import OBS_ENV, OBS_ROBOT

class DP3Policy(PreTrainedPolicy):
    """3D Diffusion Policy implementation for LeRobot."""

    config_class = DP3Config
    name = "dp3"

    def __init__(
        self,
        config: DP3Config,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Set up normalization
        self.normalize_inputs = Normalize(
            config.input_features,
            config.normalization_mapping,
            dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features,
            config.normalization_mapping,
            dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features,
            config.normalization_mapping,
            dataset_stats
        )

        # Build observation space dict for DP3Encoder
        obs_space_dict = {
            'point_cloud': config.input_features[config.point_cloud_key].shape,
            'agent_pos': config.input_features[config.agent_pos_key].shape,
        }
        if config.imagination_key in config.input_features:
            obs_space_dict['imagin_robot'] = config.input_features[config.imagination_key].shape

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.latest_gripper_proj = None

        # Initialize DP3 encoder
        self.obs_encoder = DP3Encoder(
            observation_space=obs_space_dict,
            out_channel=config.encoder_output_dim,
            pointcloud_encoder_cfg=config.pointcloud_encoder_cfg,
            use_pc_color=config.use_pc_color,
            pointnet_type=config.pointnet_type,
        )

        # Calculate dimensions
        action_dim = config.output_features["action"].shape[0]
        obs_feature_dim = self.obs_encoder.output_shape()

        # Set up UNet
        if config.obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in config.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * config.n_obs_steps
        else:
            input_dim = action_dim + obs_feature_dim
            global_cond_dim = None

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            condition_type=config.condition_type,
            use_down_condition=config.use_down_condition,
            use_mid_condition=config.use_mid_condition,
            use_up_condition=config.use_up_condition,
        )

        # Set up noise scheduler (reuse from DiffusionPolicy)
        from lerobot.common.policies.diffusion.diffusion_policy import _make_noise_scheduler
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        # For inference
        self.num_inference_steps = (
            config.num_inference_steps
            if config.num_inference_steps is not None
            else config.num_train_timesteps
        )

        # Initialize queues
        self._queues = None
        self.reset()

    def reset(self):
        """Reset observation and action queues."""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        for key in [self.config.point_cloud_key, self.config.imagination_key]:
            if key in self.config.input_features:
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare global conditioning from observations."""
        B = next(iter(batch.values())).shape[0]
        T_obs = self.config.n_obs_steps

        # Prepare encoder input
        encoder_input = {}

        # Map LeRobot keys to DP3 encoder keys
        key_mapping = {
            self.config.point_cloud_key: 'point_cloud',
            self.config.agent_pos_key: 'agent_pos',
            self.config.imagination_key: 'imagin_robot',
        }

        # Reshape batch for encoder: (B, T, ...) -> (B*T, ...)
        for lerobot_key, dp3_key in key_mapping.items():
            if lerobot_key in batch:
                data = batch[lerobot_key][:, :T_obs]
                encoder_input[dp3_key] = data.reshape(-1, *data.shape[2:])

        # Encode observations
        obs_features = self.obs_encoder(encoder_input)

        # Handle different conditioning types
        if "cross_attention" in self.config.condition_type:
            # Keep temporal dimension: (B*T, F) -> (B, T, F)
            global_cond = obs_features.reshape(B, T_obs, -1)
        else:
            # Flatten temporal dimension: (B*T, F) -> (B, T*F)
            global_cond = obs_features.reshape(B, -1)

        return global_cond

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action given observations."""
        batch = self.normalize_inputs(batch)

        # Populate queues
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # Stack observations from queues
            batch_queued = {
                k: torch.stack(list(self._queues[k]), dim=1)
                for k in batch if k in self._queues
            }

            # Get global conditioning
            global_cond = self._prepare_global_conditioning(batch_queued)

            # Generate actions
            actions = self.conditional_sample(
                batch_size=1,
                global_cond=global_cond,
            )

            # Unnormalize
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # Add to queue
            self._queues["action"].extend(actions.transpose(0, 1))

        return self._queues["action"].popleft()

    def conditional_sample(self, batch_size: int, global_cond: torch.Tensor) -> torch.Tensor:
        """Run the diffusion sampling process."""
        device = self.model.device
        dtype = self.model.dtype

        # Initialize random trajectory
        shape = (batch_size, self.config.horizon, self.config.output_features["action"].shape[0])
        trajectory = torch.randn(shape, device=device, dtype=dtype)

        # Set timesteps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Diffusion loop
        for t in self.noise_scheduler.timesteps:
            # Model prediction
            model_output = self.model(
                sample=trajectory,
                timestep=t,
                global_cond=global_cond
            )

            # Scheduler step
            trajectory = self.noise_scheduler.step(
                model_output, t, trajectory
            ).prev_sample

        # Extract action steps
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return trajectory[:, start:end]

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        """Compute loss for training."""
        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Get global conditioning
        global_cond = self._prepare_global_conditioning(batch)

        # Get actions
        actions = batch["action"]
        B, T = actions.shape[:2]

        # Prepare trajectory based on conditioning type
        if self.config.obs_as_global_cond:
            trajectory = actions
        else:
            # Would need to concatenate observations - not typically used in DP3
            raise NotImplementedError("DP3 typically uses obs_as_global_cond=True")

        # Sample noise
        noise = torch.randn_like(trajectory)

        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=trajectory.device
        ).long()

        # Add noise
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        # Model prediction
        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            global_cond=global_cond
        )

        # Compute target based on prediction type
        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")

        # MSE loss
        loss = nn.functional.mse_loss(pred, target)

        return loss, None
