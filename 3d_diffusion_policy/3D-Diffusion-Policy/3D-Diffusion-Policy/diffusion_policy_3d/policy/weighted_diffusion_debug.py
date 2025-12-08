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
from diffusion_policy_3d.model.diffusion.models_for_weighted_diffusion.create_weighted_diffusion_model import create_weighted_diffusion_model

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.model.vision.act3d_encoder import Act3dEncoder
from diffusion_policy_3d.model.vision.pointnet2_utils import PointNet2, PointNet2_no_batch_norm, PointNet2_super_no_batch_norm
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.network_helper import replace_bn_with_gn
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.model.diffusion.conditional_unet1d_4_weighted_diffusion import WeightsModel

from diffusion_policy_3d.model.vision.layers import RelativeCrossAttentionModule


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
            down_dims=(128, 128, 128),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True, # true
            use_mid_condition=True, # true
            use_up_condition=True, # true
            pointnet_type="pointnet",
            prediction_target='action',
            noise_model_type_weighted_diffusion='unet',
            diffusion_attn_embed_dim=120,
            normalize_action=True, # [Chialiang] can remove normilizer for action
            # parameters passed to step
            load_pretrained_pointnet=False,
            use_normalization=False,
            weight_loss_coef=1,
            num_classes_encoder = 128,
            model_invariant = True,

            **kwargs
        ):
        cprint("Using WeightedDiffusion", "yellow")
        super().__init__()
        print("USE NORMALIZATION !!!!!!!!!!!!!!!!!!!!!!!!!!!", use_normalization) 
        self.condition_type = condition_type
        self.prediction_target = prediction_target
        self.normalize_action = normalize_action
        self.load_pretrained_pointnet = load_pretrained_pointnet
        self.weight_loss_coef = weight_loss_coef
        self.model_invariant = model_invariant
        #self.noise_model_type = "pointnet2"
        self.input_channels = (num_classes_encoder-1) + 1 + horizon * 12 #(127 + 1 + 48) 
        
        #import pdb; pdb.set_trace()
        action_shape = shape_meta[self.prediction_target]['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        input_dim = action_dim

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        if self.load_pretrained_pointnet:
            assert prediction_target == "goal_gripper_pcd", f"Unsupported prediction target {prediction_target}"
            cprint(f"PointNet2 encoder output dim: 13!!!!!", "yellow")
            encoder_output_dim = 13
            obs_encoder_pointnet = PointNet2(num_classes=encoder_output_dim)
            self.obs_encoder_pointnet = obs_encoder_pointnet
            self.encoder_output_dim = encoder_output_dim

        #self.noise_model_type = noise_model_type
        self.noise_model_type = noise_model_type_weighted_diffusion
        if self.noise_model_type == "unet":
            if not use_normalization:
                obs_separate_encoder_pointnet = PointNet2_no_batch_norm(num_classes=129)
            else:
                obs_separate_encoder_pointnet = PointNet2(num_classes=129)
            self.separate_obs_encoder_pointnet = obs_separate_encoder_pointnet  
            self.separate_encoder_output_dim = 128
        else:
            if not use_normalization:
                obs_separate_encoder_pointnet = PointNet2_no_batch_norm(num_classes=num_classes_encoder)
            else:
                #import pdb; pdb.set_trace()
                obs_separate_encoder_pointnet = PointNet2(num_classes=num_classes_encoder)
            self.separate_obs_encoder_pointnet = obs_separate_encoder_pointnet  
            self.separate_encoder_output_dim = num_classes_encoder - 1

        obs_feature_dim = encoder_output_dim
        cprint("Using down_dims= (128, 128, 128)", "yellow")
        '''model = ConditionalUnet1D(
            input_dim=input_dim, # 12
            global_cond_dim=encoder_output_dim-1 + self.separate_encoder_output_dim if self.load_pretrained_pointnet else self.separate_encoder_output_dim, # for every point, use its own feature
            diffusion_step_embed_dim=diffusion_step_embed_dim, 
            down_dims=(128,128,128),
            kernel_size=kernel_size,
            n_groups=n_groups,
            use_group_norm=use_normalization,
        )'''
        #import pdb; pdb.set_trace()
        model = create_weighted_diffusion_model(model_invariant=self.model_invariant, model_type=self.noise_model_type, input_channels=self.input_channels, device=self.device, input_dim=input_dim, encoder_output_dim=encoder_output_dim, separate_encoder_output_dim=self.separate_encoder_output_dim, load_pretrained_pointnet=self.load_pretrained_pointnet, diffusion_step_embed_dim=diffusion_step_embed_dim, kernel_size=kernel_size, use_normalization=use_normalization, n_groups=n_groups)
        global_cond_dim = 0

        #model = replace_bn_with_gn(model)
        # self.obs_encoder_pointnet = replace_bn_with_gn(self.obs_encoder_pointnet, features_per_group=4)
        if self.load_pretrained_pointnet:
            self.obs_encoder_pointnet.load_state_dict(torch.load("/project_data/held/ziyuw2/Robogen-sim2real/test_PointNet2/exps/pointnet2_large_2024-09-19_use_75_episodes_300-obj/model_39.pth"))
            self.obs_encoder_pointnet.eval()
            for param in self.obs_encoder_pointnet.parameters():
                param.requires_grad = False

        self.model = model
        # self.weights_model = weights_model
        self.obs_encoder = None
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
            generator=None, batch_size = 1, num_scene_points = 4504, 
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
            #import pdb; pdb.set_trace()
            trajectory[condition_mask] = condition_data[condition_mask]

            #pred = self.model(sample=noisy_trajectory, timestep=timesteps, local_cond=None, global_cond=global_cond) # B*(N+4), T, 12
            if self.noise_model_type == "unet":
                model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond, **kwargs)
            else:
                trajectory_use = trajectory.reshape(batch_size, -1, num_scene_points)
                timesteps = t.repeat(batch_size*num_scene_points).to(condition_data.device)
                timesteps = timesteps.reshape(batch_size,-1, num_scene_points)
                point_net_input = torch.cat((global_cond, timesteps, trajectory_use), dim = 1).to(condition_data.device)
                model_output = self.model(xyz = point_net_input) 
                model_output = model_output.reshape(batch_size*num_scene_points, -1, self.action_dim) # B*(N+4), T, 12

            
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], pdb=False) -> Dict[str, torch.Tensor]:
        
        if pdb:
            import pdb; pdb.set_trace()
        nobs = obs_dict
        value = next(iter(nobs.values()))
        batch_size, To = value.shape[:2]
        horizon = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
        this_n_point_cloud = this_n_point_cloud[..., :3] # B, Do, N, 3
        diffuse_obj_point_cloud = this_n_point_cloud[:, -1, :, :] # B, N, 3
        diffuse_gripper_point = this_nobs['gripper_pcd'].reshape(batch_size,-1, *this_nobs['gripper_pcd'].shape[1:])
        diffuse_gripper_point = diffuse_gripper_point[..., :3]
        diffuse_gripper_point = diffuse_gripper_point[:, -1, :, :] # B, 4, 3

        diffuse_point_cloud = torch.cat([diffuse_obj_point_cloud, diffuse_gripper_point], dim=1) # B, N+4, 3
        num_scene_points = diffuse_point_cloud.shape[1]
        
        action = obs_dict[self.prediction_target]
        action = einops.rearrange(action, 'b t n d -> b (t n) d')
        action = action.unsqueeze(1).repeat(1, diffuse_point_cloud.shape[1], 1, 1) # B, N+4, T*4, 3
        diffuse_point_cloud_ = diffuse_point_cloud.unsqueeze(2).repeat(1, 1, action.shape[2], 1) # B, N, T*4, 3
        displacement = action - diffuse_point_cloud_ # B, N+4, T*4, 3
        displacement = displacement.reshape(batch_size, num_scene_points, self.horizon, 4, 3) # B, N, T, 4, 3
        trajectory = displacement.reshape(batch_size*num_scene_points, self.horizon, 4, 3) # B*(N+4), T, 4, 3
        trajectory = trajectory.reshape(batch_size*num_scene_points, self.horizon, -1) # B*(N+4), T, 12


        if self.load_pretrained_pointnet:
            global_cond = self.obs_encoder_pointnet(diffuse_point_cloud.permute(0, 2, 1)) # B, N+4, 13
            
            if pdb:
                import pdb; pdb.set_trace()
                
            
            weights = global_cond[:, :, -1] # B, N+4
            global_cond = global_cond[:, :, :-1]
            global_cond = global_cond.reshape(batch_size*num_scene_points, -1)
            
        
        
        ### add separate global conditioning
        separate_global_cond = self.separate_obs_encoder_pointnet(diffuse_point_cloud.permute(0, 2, 1))
        if self.load_pretrained_pointnet:
            separate_global_cond = separate_global_cond.reshape(batch_size*num_scene_points, -1)
            global_cond = torch.cat([global_cond, separate_global_cond], dim=-1)
        else:
            global_cond = separate_global_cond[:, :, :-1].reshape(batch_size*num_scene_points, -1)
            weights = separate_global_cond[:, :, -1]

    
        if self.noise_model_type != "unet":
            global_cond = global_cond.reshape(batch_size, -1, num_scene_points)
        ###
        
        if pdb:
            import pdb; pdb.set_trace()
        if self.noise_model_type == "unet":
            cond_data = torch.zeros(batch_size*num_scene_points, horizon, 12, device=value.device) # B*(N+4), T, 12
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            '''cond_data = torch.zeros(batch_size, horizon*12, num_scene_points, device=value.device)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)'''
            cond_data = torch.zeros(batch_size*num_scene_points, horizon, 12, device=value.device) # B*(N+4), T, 12
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        denoised_trajectory = self.conditional_sample(condition_data=cond_data, condition_mask=cond_mask, local_cond=None, global_cond=global_cond, batch_size=batch_size, num_scene_points=num_scene_points, **self.kwargs) # B*N, T, 12


        denoised_trajectory = denoised_trajectory.reshape(batch_size, num_scene_points, horizon, 12)
        denoised_trajectory = denoised_trajectory.reshape(batch_size, num_scene_points, horizon, 4, 3) # B, (N+4), T, 4, 3
        pred_goal_gripper_pcd = denoised_trajectory + diffuse_point_cloud.unsqueeze(2).unsqueeze(3).repeat(1, 1, horizon, 4, 1) # B, (N+4), T, 4, 3
        pred_goal_gripper_pcd = pred_goal_gripper_pcd.reshape(batch_size, num_scene_points, -1) # B, (N+4)`, T*4*3

        weights = torch.nn.functional.softmax(weights, dim=1) # B, N+4
        pred_goal_gripper = (pred_goal_gripper_pcd * weights.unsqueeze(-1)).sum(dim=1) # B, T*4*3
        pred_goal_gripper = pred_goal_gripper.reshape(batch_size, horizon, 4, 3) # B, T, 4, 3
        pred_goal_gripper = pred_goal_gripper.reshape(batch_size, horizon, -1) # B, T, 12
        # import pdb; pdb.set_trace()
        gt_goal_gripper = obs_dict[self.prediction_target]
        gt_goal_gripper = gt_goal_gripper.flatten(start_dim=2)
        print("after weighted average error: ", F.mse_loss(gt_goal_gripper, pred_goal_gripper).item())
        
        if pdb:
            import pdb; pdb.set_trace()

        action_pred = pred_goal_gripper[...,:Da]
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
            # "outputs": outputs,
            # 'error': error
        }
        
        return result


    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        
    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_loss(self, batch):
        obs_dict = batch['obs']
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
        prediction_horizon = nactions.shape[1]

        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))

        
        this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
        this_n_point_cloud = this_n_point_cloud[..., :3] # B, Do, N, 3
        
        diffuse_obj_point_cloud = this_n_point_cloud[:, -1, :, :] # B, N, 3
        diffuse_gripper_point = this_nobs['gripper_pcd'].reshape(batch_size,-1, *this_nobs['gripper_pcd'].shape[1:])
        diffuse_gripper_point = diffuse_gripper_point[..., :3]
        diffuse_gripper_point = diffuse_gripper_point[:, -1, :, :] # B, 4, 3

        diffuse_point_cloud = torch.cat([diffuse_obj_point_cloud, diffuse_gripper_point], dim=1) # B, N+4, 3
        num_scene_points = diffuse_point_cloud.shape[1]

        if self.load_pretrained_pointnet:
            #import pdb; pdb.set_trace()
            global_cond = self.obs_encoder_pointnet(diffuse_point_cloud.permute(0, 2, 1)) # B, N+4, 13
            #import pdb; pdb.set_trace()
            ### perform an infer using the pointnet++ output here
            # import pdb; pdb.set_trace()
            # outputs = global_cond
            # weights = outputs[:, :, -1] # B, N
            # outputs = outputs[:, :, :-1] # B, N, 12

            # B, N, _ = outputs.shape
            # outputs = outputs.view(B, N, 4, 3)
                
            # inputs = diffuse_point_cloud
            # outputs = outputs + inputs.unsqueeze(2)
            # weights = torch.nn.functional.softmax(weights, dim=1)
            # outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
            # outputs = outputs.sum(dim=1)
            
            # labels = nactions[:, 0]
            # outputs = outputs.reshape(B, -1)
            # error = ((outputs - labels) ** 2).mean()
            # cprint("in computing loss, pretrained pointnet++ prediction loss is {}".format(error), 'green')
            ##### 
            
            weights = global_cond[:, :, -1] # B, N+4
            global_cond = global_cond[:, :, :-1]
            global_cond = global_cond.reshape(batch_size*num_scene_points, -1) # B*(N+4), 12
        
            ### add separate global conditioning
            separate_global_cond = self.separate_obs_encoder_pointnet(diffuse_point_cloud.permute(0, 2, 1))
            separate_global_cond = separate_global_cond.reshape(batch_size*num_scene_points, -1)
            global_cond = torch.cat([global_cond, separate_global_cond], dim=-1)
            ### 
        else:
            #import pdb; pdb.set_trace()
            separate_global_cond = self.separate_obs_encoder_pointnet(diffuse_point_cloud.permute(0, 2, 1))
            #import pdb; pdb.set_trace()
            weights = separate_global_cond[:, :, -1]
            if self.noise_model_type == "unet":
                global_cond = separate_global_cond[:, :, :-1].reshape(batch_size*num_scene_points, -1)
            else:
                #import pdb; pdb.set_trace()
                global_cond = separate_global_cond[:, :, :-1].reshape(batch_size, -1, num_scene_points)
        
        ori_diffuse_point_cloud = diffuse_point_cloud.permute(0, 2 ,1)
        action = batch['obs'][self.prediction_target]
        action = einops.rearrange(action, 'b t n d -> b (t n) d')
        action = action.unsqueeze(1).repeat(1, diffuse_point_cloud.shape[1], 1, 1) # B, N+4, T*4, 3
        diffuse_point_cloud = diffuse_point_cloud.unsqueeze(2).repeat(1, 1, action.shape[2], 1) # B, N, T*4, 3
        displacement = action - diffuse_point_cloud # B, N+4, T*4, 3
        displacement = displacement.reshape(batch_size, num_scene_points, horizon, 4, 3) # B, N, T, 4, 3
        trajectory = displacement.reshape(batch_size*num_scene_points, horizon, 4, 3) # B*(N+4), T, 4, 3
        trajectory = trajectory.reshape(batch_size*num_scene_points, horizon, -1) # B*(N+4), T, 12
        
        cond_data = trajectory
        condition_mask = self.mask_generator(trajectory.shape)
        loss_mask = ~condition_mask
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        
        if self.noise_model_type == "unet":
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long()
        else:
            #import pdb; pdb.set_trace()
            random_number = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (1,), device=trajectory.device)
            timesteps = random_number.repeat(bsz)
            timesteps = timesteps.reshape(batch_size,-1, num_scene_points)

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        if self.noise_model_type == "unet":
            #import pdb; pdb.set_trace()
            pred = self.model(sample=noisy_trajectory, timestep=timesteps, local_cond=None, global_cond=global_cond) # B*(N+4), T, 12
        else:
            #import pdb; pdb.set_trace()
            noisy_trajectory = noisy_trajectory.reshape(batch_size, -1, num_scene_points)
            point_net_input = torch.cat((global_cond, timesteps, noisy_trajectory), dim = 1)
            pred = self.model(xyz = point_net_input) 
            pred = pred.reshape(batch_size*num_scene_points, -1, self.action_dim) # B*(N+4), T, 12
        #import pdb; pdb.set_trace()
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
        #import pdb; pdb.set_trace()
        diffusion_loss = F.mse_loss(pred, target, reduction='none')
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
        diffusion_loss = reduce(diffusion_loss, 'b ... -> b (...)', 'mean')
        diffusion_loss = diffusion_loss.mean()
        #import pdb; pdb.set_trace()
        if pred_type == 'epsilon':
            #import pdb; pdb.set_trace()
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(trajectory.device)
            timesteps = timesteps.to(trajectory.device)
            #import pdb; pdb.set_trace()
            sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            while len(sqrt_alpha_prod.shape) < len(noisy_trajectory.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            #import pdb; pdb.set_trace()
            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_trajectory.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            #import pdb; pdb.set_trace()
            # known noisy_trajectory = sqrt_alpha_prod * denoised_trajectory + sqrt_one_minus_alpha_prod * noise
            if self.noise_model_type != "unet":
                noisy_trajectory = noisy_trajectory.reshape(batch_size*num_scene_points, -1, self.action_dim)
            #import pdb; pdb.set_trace()
            denoised_trajectory = (noisy_trajectory - sqrt_one_minus_alpha_prod * pred) / sqrt_alpha_prod # B*N, T, 12
        elif pred_type == 'sample':
            denoised_trajectory = pred # B*N, T, 12
        #import pdb; pdb.set_trace()
        denoised_trajectory = denoised_trajectory.reshape(batch_size, num_scene_points, horizon, 12)
        denoised_trajectory = denoised_trajectory.reshape(batch_size, num_scene_points, horizon, 4, 3) # B, N+4, T, 4, 3
        pred_goal_gripper_pcd = denoised_trajectory + diffuse_point_cloud.reshape(batch_size, num_scene_points, horizon, 4, 3) # B, N+4, T, 4, 3
        #import pdb; pdb.set_trace()
        pred_goal_gripper_pcd = pred_goal_gripper_pcd.reshape(batch_size, num_scene_points, -1) # B, N+4, T*4*3
        #import pdb; pdb.set_trace()
        weights = torch.nn.functional.softmax(weights, dim=1) # B, N+4
        pred_goal_gripper = (pred_goal_gripper_pcd * weights.unsqueeze(-1)).sum(dim=1) # B, T*4*3
        pred_goal_gripper = pred_goal_gripper.reshape(batch_size, prediction_horizon, 4, 3) # B, T, 4, 3
        label = batch['obs'][self.prediction_target] # B, T, 4, 3
        weighting_loss = F.mse_loss(pred_goal_gripper, label)
        #import pdb; pdb.set_trace()
        loss = diffusion_loss + weighting_loss * self.weight_loss_coef
        loss = loss.mean()
        #import pdb; pdb.set_trace()
        loss_dict = {
            'bc_loss': loss.item(),
            'diffusion_loss': diffusion_loss.item(),
            'weighting_loss': weighting_loss.item() * self.weight_loss_coef,
        }
        return loss, loss_dict
