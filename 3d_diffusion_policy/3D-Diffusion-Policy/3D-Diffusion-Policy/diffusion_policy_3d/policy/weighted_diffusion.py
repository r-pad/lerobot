from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import einops

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.model.vision.act3d_encoder import Act3dEncoder
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.network_helper import replace_bn_with_gn
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.diffusion.conditional_unet1d_4_weighted_diffusion import WeightsModel
# from diffusion_policy_3d.model.diffusion.conditional_unet1d_4_weighted_diffusion import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator

class WeightedDiffusion(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True, # true
            use_mid_condition=True, # true
            use_up_condition=True, # true
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            use_state=True,
            encoder_type='pointnet',
            act3d_encoder_cfg=None,
            prediction_target='action',
            noise_model_type='unet',
            diffusion_attn_embed_dim=120,
            normalize_action=True, # [Chialiang] can remove normilizer for action
            # parameters passed to step
            weight_loss_weight=1.0,
            diffusion_loss_weight=1.0,
            **kwargs
        ):
        cprint("Using WeightedDiffusion", "yellow")
        super().__init__()

        self.condition_type = condition_type
        self.prediction_target = prediction_target
        self.normalize_action = normalize_action
        self.encoder_type = encoder_type
        self.diffusion_loss_weight = diffusion_loss_weight
        self.weight_loss_weight = weight_loss_weight

        assert prediction_target == "goal_gripper_pcd", f"Unsupported prediction target {prediction_target}"

        action_shape = shape_meta[self.prediction_target]['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        if self.encoder_type=="dp3":
            obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                use_state=use_state
                                                )
            # create diffusion model
            obs_feature_dim = obs_encoder.output_shape()
            input_dim = action_dim + obs_feature_dim
            global_cond_dim = None
            if obs_as_global_cond:
                input_dim = action_dim
                if "cross_attention" in self.condition_type:
                    global_cond_dim = obs_feature_dim
                else:
                    global_cond_dim = obs_feature_dim * n_obs_steps
                    
            self.use_pc_color = use_pc_color
            self.pointnet_type = pointnet_type
            cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
            cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")
        elif self.encoder_type == 'act3d':
            obs_encoder = Act3dEncoder(**act3d_encoder_cfg, encoder_output_dim=encoder_output_dim, 
                                       observation_space=obs_dict)

            obs_feature_dim = obs_encoder.output_shape()
            input_dim = action_dim + obs_feature_dim
            global_cond_dim = None
            if obs_as_global_cond:
                input_dim = action_dim
                if "cross_attention" in self.condition_type:
                    global_cond_dim = obs_feature_dim
                else:
                    global_cond_dim = obs_feature_dim * n_obs_steps
        else:
            raise NotImplementedError(f"Unsupported encoder type {self.encoder_type}")

        self.encoder_output_dim = encoder_output_dim
        self.noise_model_type = noise_model_type

        model = ConditionalUnet1D(
            input_dim=input_dim, # 12
            global_cond_dim=global_cond_dim + encoder_output_dim, # for every point, use its own feature too
            diffusion_step_embed_dim=diffusion_step_embed_dim, 
            down_dims=(256,256,256),
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        weights_model = WeightsModel(
            input_dim=input_dim * horizon + global_cond_dim + encoder_output_dim, # for every point, use current noise displacement, global feature, own feature to predict weight
        )

        if self.encoder_type == "act3d":
            model = replace_bn_with_gn(model)
            weights_model = replace_bn_with_gn(weights_model)

        self.model = model
        self.weights_model = weights_model
        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler

        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print_params(self)

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

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond, **kwargs)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = obs_dict
        value = next(iter(nobs.values()))
        batch_size, To = value.shape[:2]
        horizon = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        

        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # B*To, 304

        if "cross_attention" in self.condition_type:
            # treat as a sequence
            global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        else:
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1) # B, 608
        this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
        this_n_point_cloud = this_n_point_cloud[..., :3] # B, Do, N, 3

        diffuse_point_cloud = this_n_point_cloud[:, -1, :, :] # B, N, 3
        scene_points, scene_features = self.obs_encoder.get_rgb_features()
        scene_points = scene_points.reshape(batch_size, self.n_obs_steps, -1, 3)
        num_scene_points = scene_points.shape[2]
        scene_features = scene_features.reshape(num_scene_points, batch_size, self.n_obs_steps, self.encoder_output_dim)
        scene_features = scene_features.permute(1, 2, 0, 3) # B, Do, N, 60
        scene_features = scene_features[:, -1, :, :] # B, N, 

        global_cond = global_cond.unsqueeze(1).repeat(1, scene_features.shape[1], 1) # B, N, 608
        global_cond = torch.cat([global_cond, scene_features], dim=-1) # B, N, 608 + 60
        global_cond = global_cond.reshape(batch_size*num_scene_points, -1) # B*N, 608 + 60

        cond_data = torch.zeros(batch_size*num_scene_points, horizon, 12, device=value.device) # B*N, T, 12
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        denoised_trajectory = self.conditional_sample(
            condition_data=cond_data, condition_mask=cond_mask,
            local_cond=None, global_cond=global_cond,
            **self.kwargs) # B*N, T, 12
        denoised_trajectory = denoised_trajectory.reshape(batch_size, num_scene_points, horizon, 12)
        denoised_trajectory = denoised_trajectory.reshape(batch_size, num_scene_points, horizon, 4, 3) # B, N, T, 4, 3
        pred_goal_gripper_pcd = denoised_trajectory + diffuse_point_cloud.unsqueeze(2).unsqueeze(3).repeat(1, 1, horizon, 4, 1) # B, N, T, 4, 3
        pred_goal_gripper_pcd = pred_goal_gripper_pcd.reshape(batch_size, num_scene_points, -1) # B, N, T*4*3

        weights = torch.cat([pred_goal_gripper_pcd, global_cond.reshape(batch_size, num_scene_points, -1)], dim=-1) # B, N, T*4*3 + 608 + 60
        weights = self.weights_model(weights)
        weights = weights[:, :, 0]
        weights = torch.nn.functional.softmax(weights, dim=1) # B, N
        pred_goal_gripper = (pred_goal_gripper_pcd * weights.unsqueeze(-1)).sum(dim=1) # B, T*4*3
        pred_goal_gripper = pred_goal_gripper.reshape(batch_size, horizon, 4, 3) # B, T, 4, 3
        pred_goal_gripper = pred_goal_gripper.reshape(batch_size, horizon, -1) # B, T, 12

        action_pred = pred_goal_gripper[...,:Da]
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result
        


    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        
    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_loss(self, batch):
        nactions = batch['obs'][self.prediction_target].flatten(start_dim=2)
        nobs = batch['obs']

        # action: B, T, 10
        # obs
        # point_cloud: B, T, N, 3
        # agent_pos: B, T, 10
        # gripper_pcd: B, T, 4, 3
        # displacement_gripper_to_object: B, T, 4, 3
        # goal_gripper_pcd: B, T, 4, 3

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)

        if "cross_attention" in self.condition_type:
            # treat as a sequence
            global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        else:
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1) # B, 608
        # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
        this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
        this_n_point_cloud = this_n_point_cloud[..., :3] # B, Do, N, 3

        diffuse_point_cloud = this_n_point_cloud[:, -1, :, :] # B, N, 3
        scene_points, scene_features = self.obs_encoder.get_rgb_features()
        scene_points = scene_points.reshape(batch_size, self.n_obs_steps, -1, 3)
        num_scene_points = scene_points.shape[2]
        scene_features = scene_features.reshape(num_scene_points, batch_size, self.n_obs_steps, self.encoder_output_dim)
        scene_features = scene_features.permute(1, 2, 0, 3) # B, Do, N, 60
        scene_features = scene_features[:, -1, :, :] # B, N, 60

        global_cond = global_cond.unsqueeze(1).repeat(1, scene_features.shape[1], 1) # B, N, 608
        global_cond = torch.cat([global_cond, scene_features], dim=-1) # B, N, 608 + 60

        action = batch['obs'][self.prediction_target] # B, T, 4, 3
        # calculate the displacement from diffuse_point_cloud to action
        action = einops.rearrange(action, 'b t n d -> b (t n) d') # B, T*4, 3
        diffuse_point_cloud = diffuse_point_cloud.unsqueeze(2).repeat(1, 1, action.shape[1], 1) # B, N, T*4, 3
        action = action.unsqueeze(1).repeat(1, diffuse_point_cloud.shape[1], 1, 1) # B, N, T*4, 3
        displacement = action - diffuse_point_cloud # B, N, T*4, 3
        displacement = displacement.reshape(batch_size, num_scene_points, horizon, 4, 3) # B, N, T, 4, 3

        trajectory = displacement.reshape(batch_size*num_scene_points, horizon, 4, 3) # B*N, T, 4, 3
        trajectory = trajectory.reshape(batch_size*num_scene_points, horizon, -1) # B*N, T, 12
        global_cond = global_cond.reshape(batch_size*num_scene_points, -1) # B*N, 608 + 60
        cond_data = trajectory

        condition_mask = self.mask_generator(trajectory.shape)
        loss_mask = ~condition_mask
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long()

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        
        pred = self.model(sample=noisy_trajectory, timestep=timesteps, local_cond=None, global_cond=global_cond) # B*N, T, 12

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            v_t = self.noise_scheduler.get_velocity(sample=trajectory, noise=noise, timesteps=timesteps)
            target = v_t
        else:
            raise NotImplementedError(f"Prediction type {pred_type} not implemented")

        diffusion_loss = F.mse_loss(pred, target, reduction='none')
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
        diffusion_loss = reduce(diffusion_loss, 'b ... -> b (...)', 'mean')
        diffusion_loss = diffusion_loss.mean()

        # denoised_trajectory = self.noise_scheduler.step(pred, timesteps, noisy_trajectory)

        if pred_type == 'epsilon':
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(trajectory.device)
            timesteps = timesteps.to(trajectory.device)

            sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            while len(sqrt_alpha_prod.shape) < len(noisy_trajectory.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_trajectory.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

            # known noisy_trajectory = sqrt_alpha_prod * denoised_trajectory + sqrt_one_minus_alpha_prod * noise
            denoised_trajectory = (noisy_trajectory - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod # B*N, T, 12
        elif pred_type == 'sample':
            denoised_trajectory = pred # B*N, T, 12

        denoised_trajectory = denoised_trajectory.reshape(batch_size, num_scene_points, horizon, 12)
        denoised_trajectory = denoised_trajectory.reshape(batch_size, num_scene_points, horizon, 4, 3) # B, N, T, 4, 3
        pred_goal_gripper_pcd = denoised_trajectory + diffuse_point_cloud.reshape(batch_size, num_scene_points, horizon, 4, 3) # B, N, T, 4, 3

        pred_goal_gripper_pcd = pred_goal_gripper_pcd.reshape(batch_size, num_scene_points, -1) # B, N, T*4*3
        weights = torch.cat([pred_goal_gripper_pcd, global_cond.reshape(batch_size, num_scene_points, -1)], dim=-1) # B, N, T*4*3 + 608 + 60
        weights = self.weights_model(weights)
        weights = weights[:, :, 0]
        weights = torch.nn.functional.softmax(weights, dim=1) # B, N
        pred_goal_gripper = (pred_goal_gripper_pcd * weights.unsqueeze(-1)).sum(dim=1) # B, T*4*3
        pred_goal_gripper = pred_goal_gripper.reshape(batch_size, horizon, 4, 3) # B, T, 4, 3
        label = batch['obs'][self.prediction_target] # B, T, 4, 3
        weighting_loss = F.mse_loss(pred_goal_gripper, label)

        loss = self.diffusion_loss_weight * diffusion_loss + self.weight_loss_weight * weighting_loss
        loss = loss.mean()

        loss_dict = {
            'bc_loss': loss.item(),
            'diffusion_loss': diffusion_loss.item() * self.diffusion_loss_weight,
            'weighting_loss': weighting_loss.item() * self.weight_loss_weight,
        }
        return loss, loss_dict
        