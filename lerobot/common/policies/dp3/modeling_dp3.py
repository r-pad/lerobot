import math
from collections import deque
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn
from itertools import chain
from einops import rearrange, reduce

from lerobot.common.constants import OBS_ENV, OBS_ROBOT
from lerobot.common.policies.dp3.configuration_dp3 import DP3Config
from lerobot.common.policies.dp3.pointnet_extractor import DP3Encoder
from lerobot.common.policies.dp3.conditional_unet1d import ConditionalUnet1D
from lerobot.common.policies.dp3.mask_generator import LowdimMaskGenerator
from lerobot.common.policies.dp3.pytorch_util import dict_apply
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy

from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.common.policies.high_level.high_level_wrapper import HighLevelWrapper, HighLevelConfig, get_siglip_text_embedding
from transformers import AutoModel, AutoProcessor


class DP3Policy(PreTrainedPolicy):
    """
    Modified 3D Diffusion Policy from Articubot
    """
    config_class = DP3Config
    name = "dp3"

    def __init__(
        self,
        config: DP3Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.robot_adapter = config.get_robot_adapter()
        self.obs_key = self.robot_adapter.get_obs_key()
        self.act_key = self.robot_adapter.get_act_key()

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None
        self.latest_gripper_proj = {}
        # For relative actions: save reference EEF pose for the current action chunk
        self._relative_action_reference_eef = None

        # instantiate DP3 model
        # TODO: obs_dict, action_dim
        obs_dict = {
            'point_cloud': (4500, 3),
            'imagin_robot': (4, 3),
            'goal_gripper_pcd': (4, 3),
            'agent_pos': dataset_stats['observation.state']['mean'].shape
        }
        action_dim = dataset_stats[self.act_key]['mean'].shape[0]
        pointcloud_encoder_cfg = dict(in_channels=config.in_channels,
                                    out_channels=config.out_channels,
                                    use_layernorm=config.use_layernorm,
                                    final_norm=config.final_norm,
                                    normal_channel=config.normal_channel)
        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                img_crop_shape=config.crop_shape,
                                out_channel=config.encoder_output_dim,
                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                use_pc_color=config.use_pc_color,
                                pointnet_type=config.pointnet_type,
                                enable_goal_conditioning=config.enable_goal_conditioning,
                                eef_points=config.eef_points,
                                embedding_type=config.embedding_type
                                )
        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if config.obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in config.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * config.n_obs_steps
        self.use_pc_color = config.use_pc_color
        self.pointnet_type = config.pointnet_type
        self.aloha_gripper_idx = torch.tensor([6, 197, 174]) # Handpicked idxs for the aloha
        model = ConditionalUnet1D(
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
        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if config.obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=config.n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        if config.noise_scheduler_type == "DDPM":
            self.noise_scheduler = DDPMScheduler()
        elif config.noise_scheduler_type == "DDIM":
            self.noise_scheduler = DDIMScheduler()
        else:
            raise NotImplementedError

        if self.config.enable_goal_conditioning:
            hl_config = HighLevelConfig(
                model_type=self.config.hl_model_type,
                run_id=self.config.hl_run_id,
                entity=self.config.hl_entity,
                project=self.config.hl_project,
                checkpoint_type=self.config.hl_checkpoint_type,
                max_depth=self.config.hl_max_depth,
                num_points=self.config.hl_num_points,
                in_channels=self.config.hl_in_channels,
                use_gripper_pcd=self.config.hl_use_gripper_pcd,
                use_text_embedding=self.config.hl_use_text_embedding,
                use_dual_head=self.config.hl_use_dual_head,
                use_rgb=self.config.hl_use_rgb,
                use_gemini=self.config.hl_use_gemini,
                is_gmm=self.config.hl_is_gmm,
                dino_model=self.config.hl_dino_model,
                calibration_json=self.config.hl_calibration_json,
                use_fourier_pe=self.config.hl_use_fourier_pe,
                fourier_num_frequencies=self.config.hl_fourier_num_frequencies,
                fourier_include_input=self.config.hl_fourier_include_input,
                num_transformer_layers=self.config.hl_num_transformer_layers,
                dropout=self.config.hl_dropout,
                use_source_token=self.config.hl_use_source_token,
                use_gripper_token=self.config.hl_use_gripper_token,
            )
            self.high_level = HighLevelWrapper(hl_config)

        self.renderer = None
        self.reset()

    def get_optim_params(self) -> dict:
        params = chain(
            self.obs_encoder.parameters(),
            self.model.parameters()
        )
        return params

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            self.obs_key: deque(maxlen=self.config.n_obs_steps),
            self.act_key: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        state = batch['observation.state']
        current_obs = batch[self.obs_key]

        # For now, only normalize outputs
        # batch = self.normalize_inputs(batch)
        if self.config.use_text_embedding:
            text = batch['task']

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[self.act_key]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            if self.config.use_text_embedding:
                batch['task'] = text

            # Save the reference observation for this action chunk (used for relative actions)
            self._relative_action_reference_eef = current_obs.squeeze(0)  # (10,)

            # pass observation to encoder
            nobs_features = self.obs_encoder(batch)
            # TODO: B, T, Da
            if self.config.obs_as_global_cond:
                if "cross_attention" in self.config.condition_type:
                    # treat as a sequence
                    global_cond = nobs_features.reshape(B, self.config.n_obs_steps, -1)
                else:
                    # reshape back to B, Do
                    global_cond = nobs_features.reshape(B, -1)
                # empty data for action
                cond_data = torch.zeros(size=(B, T, Da))
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            else:
                raise NotImplementedError
            # run sampling
            nsample = self.conditional_sample(
                                cond_data, 
                                cond_mask,
                                global_cond=global_cond)

            # TODO(rcadene): make above methods return output dictionary?
            actions = self.unnormalize_outputs({self.act_key: actions})[self.act_key]

            self._queues[self.act_key].extend(actions.transpose(0, 1))

        action_raw = self._queues[self.act_key].popleft()
        action = self.robot_adapter.transform_action(action_raw, state, self._relative_action_reference_eef)
        action_eef = self.robot_adapter.get_eef_action(action_raw, state, self._relative_action_reference_eef)
        return action, action_eef

    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.action_space == "right_eef_relative":
            batch = self.robot_adapter.compute_relative_actions(batch)

        # For now, only normalize outputs
        # batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        batch = self.normalize_targets(batch)
        
        # compute loss
        # pre-process gripper pcds by selecting via indices
        gripper_pcds = batch['observation.points.gripper_pcds'][..., self.aloha_gripper_idx, :]
        fourth_point = (gripper_pcds[..., 0, :] + gripper_pcds[..., 1, :]) / 2.
        gripper_pcds = torch.concat([gripper_pcds, fourth_point.unsqueeze(2)], dim=-2)

        goal_gripper_pcds = batch['observation.points.goal_gripper_pcds'][..., self.aloha_gripper_idx, :]
        fourth_goal_point = (goal_gripper_pcds[..., 0, :] + goal_gripper_pcds[..., 1, :]) / 2.
        goal_gripper_pcds = torch.concat([goal_gripper_pcds, fourth_goal_point.unsqueeze(2)], dim=-2)

        nobs = {
            'agent_pos': batch['observation.state'],
            'imagin_robot': gripper_pcds,
            'goal_gripper_pcd': goal_gripper_pcds,
            'point_cloud': batch['observation.points.point_cloud'],
        }
        nactions = batch[self.act_key]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.config.obs_as_global_cond:
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.config.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.config.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.config.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            raise NotImplementedError
        
        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()


        # no output_dict so returning None
        return loss, None