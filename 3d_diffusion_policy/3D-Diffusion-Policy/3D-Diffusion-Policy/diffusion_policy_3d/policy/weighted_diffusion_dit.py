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
from torch.nn.init import xavier_uniform_, calculate_gain

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
            noise_model_type='unet',
            normalize_action=True, # [Chialiang] can remove normilizer for action
            # parameters passed to step
            load_pretrained_pointnet=False,
            use_normalization=False,
            weight_loss_coef=1,
            pointnet_encoder_output_dim=129,
            pointnet_class='pointnet2_large', 
            **kwargs
        ):
        cprint("Using WeightedDiffusion", "yellow")
        super().__init__()

        self.condition_type = condition_type
        self.prediction_target = prediction_target
        self.normalize_action = normalize_action
        self.load_pretrained_pointnet = load_pretrained_pointnet
        self.weight_loss_coef = weight_loss_coef

        

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

        self.noise_model_type = noise_model_type

        if not use_normalization:
            if pointnet_class == 'pointnet2_large':
                obs_separate_encoder_pointnet = PointNet2_no_batch_norm(num_classes=pointnet_encoder_output_dim)
            elif pointnet_class == 'pointnet2_super':
                obs_separate_encoder_pointnet = PointNet2_super_no_batch_norm(num_classes=pointnet_encoder_output_dim)
                
        else:
            obs_separate_encoder_pointnet = PointNet2(num_classes=pointnet_encoder_output_dim)
        self.separate_obs_encoder_pointnet = obs_separate_encoder_pointnet  
        self.separate_encoder_output_dim = pointnet_encoder_output_dim - 1


        obs_feature_dim = encoder_output_dim if self.load_pretrained_pointnet else self.separate_encoder_output_dim
        if self.noise_model_type == 'unet':        
            model = ConditionalUnet1D(
                input_dim=input_dim, # 12
                global_cond_dim=encoder_output_dim-1 + self.separate_encoder_output_dim if self.load_pretrained_pointnet else self.separate_encoder_output_dim, # for every point, use its own feature
                diffusion_step_embed_dim=diffusion_step_embed_dim, 
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                use_group_norm=use_normalization,
                horizon=horizon,
            )
            global_cond_dim = 0
            model = replace_bn_with_gn(model)
        elif self.noise_model_type == 'pointnet':
            model = PointNet2_no_batch_norm(num_classes=input_dim * horizon, diffusion=True, 
                                global_cond_dim=self.separate_encoder_output_dim, sample_dim=input_dim * horizon, time_embedding_dim=diffusion_step_embed_dim)
            # initialize_network(model)

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
            generator=None,
            pcd=None,
            gt_trajectory=None,
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


            if self.noise_model_type == 'unet':
                model_output = model(sample=trajectory,
                                    timestep=t, 
                                    local_cond=local_cond, global_cond=global_cond, **kwargs)
            elif self.noise_model_type == 'pointnet':
                # cond_data = torch.zeros(batch_size*num_scene_points, horizon, 12, device=value.device)
                batch_size, num_scene_points = pcd.shape[0], pcd.shape[1]
                input_trajectory = trajectory.reshape(batch_size, num_scene_points, self.horizon, -1)
                input_global_cond = global_cond.reshape(batch_size, num_scene_points, -1)
                t_input = t[None].to(condition_data.device)
                t_input = t_input.expand(batch_size * num_scene_points)   
                model_output = model.forward_diffusion(sample=input_trajectory, timestep=t_input, global_cond=input_global_cond, xyz=pcd.permute(0, 2, 1))
                model_output = model_output.reshape(batch_size * num_scene_points, self.horizon, -1)
                # import pdb; pdb.set_trace()
                
                with torch.no_grad():
                    loss = F.mse_loss(model_output, gt_trajectory)
                    print("denoising step {} loss {}".format(t, loss.item()))
                
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
                
            ### get the pretrained pointnet++ prediction
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
            
            # labels = this_nobs['goal_gripper_pcd'].reshape(B, 2, 12)[:, 0]
            # outputs = outputs.reshape(B, -1)
            # outputs = outputs.reshape(B, 4, 3)
            # labels = labels.reshape(B, 4 ,3)
            # error = torch.linalg.norm(outputs - labels, dim=-1).mean()
            # cprint("in predict action, pretrained pointnet++ prediction loss is {}".format(error), 'yellow')
            #######
            
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
        ###
        
        if pdb:
            import pdb; pdb.set_trace()

        cond_data = torch.zeros(batch_size*num_scene_points, horizon, 12, device=value.device) # B*(N+4), T, 12
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        denoised_trajectory = self.conditional_sample(condition_data=cond_data, condition_mask=cond_mask, local_cond=None, global_cond=global_cond, 
                                    pcd=diffuse_point_cloud, gt_trajectory=trajectory, **self.kwargs) # B*N, T, 12
        


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
            global_cond = self.obs_encoder_pointnet(diffuse_point_cloud.permute(0, 2, 1)) # B, N+4, 13
            
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
            separate_global_cond = self.separate_obs_encoder_pointnet(diffuse_point_cloud.permute(0, 2, 1))
            weights = separate_global_cond[:, :, -1]
            global_cond = separate_global_cond[:, :, :-1].reshape(batch_size*num_scene_points, -1)
        
        ori_diffuse_point_cloud = diffuse_point_cloud.permute(0, 2 ,1)
        action = batch['obs'][self.prediction_target]
        action = einops.rearrange(action, 'b t n d -> b (t n) d')
        action = action.unsqueeze(1).repeat(1, diffuse_point_cloud.shape[1], 1, 1) # B, N+4, T*4, 3
        diffuse_point_cloud = diffuse_point_cloud.unsqueeze(2).repeat(1, 1, action.shape[2], 1) # B, N, T*4, 3
        displacement = action - diffuse_point_cloud # B, N+4, T*4, 3
        displacement = displacement.reshape(batch_size, num_scene_points, prediction_horizon, 4, 3) # B, N, T, 4, 3
        trajectory = displacement.reshape(batch_size*num_scene_points, prediction_horizon, 4, 3) # B*(N+4), T, 4, 3
        trajectory = trajectory.reshape(batch_size*num_scene_points, prediction_horizon, -1) # B*(N+4), T, 12

        cond_data = trajectory
        condition_mask = self.mask_generator(trajectory.shape)
        loss_mask = ~condition_mask
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]


        if self.noise_model_type == 'unet':
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long()
            noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
            # import pdb; pdb.set_trace()
            pred = self.model(sample=noisy_trajectory, timestep=timesteps, local_cond=None, global_cond=global_cond) # B*(N+4), T, 12
        elif self.noise_model_type == "pointnet":
            global_cond = global_cond.reshape(batch_size, num_scene_points, -1)
            bsz = batch_size
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device).long()
            timesteps = timesteps.unsqueeze(1).repeat(1, num_scene_points).flatten()            
            noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
            noisy_trajectory = noisy_trajectory.reshape(batch_size, num_scene_points, prediction_horizon, -1)
            pred = self.model.forward_diffusion(sample=noisy_trajectory, timestep=timesteps, global_cond=global_cond, xyz=ori_diffuse_point_cloud)
            # import pdb; pdb.set_trace()
            pred = pred.reshape(batch_size * num_scene_points, prediction_horizon, -1)
            noisy_trajectory = noisy_trajectory.reshape(batch_size * num_scene_points, prediction_horizon, -1)

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
            denoised_trajectory = (noisy_trajectory - sqrt_one_minus_alpha_prod * pred) / sqrt_alpha_prod # B*N, T, 12
        elif pred_type == 'sample':
            denoised_trajectory = pred # B*N, T, 12

        denoised_trajectory = denoised_trajectory.reshape(batch_size, num_scene_points, prediction_horizon, 12)
        denoised_trajectory = denoised_trajectory.reshape(batch_size, num_scene_points, prediction_horizon, 4, 3) # B, N+4, T, 4, 3
        pred_goal_gripper_pcd = denoised_trajectory + diffuse_point_cloud.reshape(batch_size, num_scene_points, prediction_horizon, 4, 3) # B, N+4, T, 4, 3

        pred_goal_gripper_pcd = pred_goal_gripper_pcd.reshape(batch_size, num_scene_points, -1) # B, N+4, T*4*3

        weights = torch.nn.functional.softmax(weights, dim=1) # B, N+4
        pred_goal_gripper = (pred_goal_gripper_pcd * weights.unsqueeze(-1)).sum(dim=1) # B, T*4*3
        pred_goal_gripper = pred_goal_gripper.reshape(batch_size, prediction_horizon, 4, 3) # B, T, 4, 3
        label = batch['obs'][self.prediction_target] # B, T, 4, 3
        weighting_loss = F.mse_loss(pred_goal_gripper, label)

        loss = diffusion_loss + weighting_loss * self.weight_loss_coef
        loss = loss.mean()

        loss_dict = {
            'bc_loss': loss.item(),
            'diffusion_loss': diffusion_loss.item(),
            'weighting_loss': weighting_loss.item() * self.weight_loss_coef,
        }
        return loss, loss_dict
    
def custom_weight_init(module, scale=0.01):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
        # Xavier initialization with a custom scale
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
        gain = calculate_gain('relu')  # Assuming ReLU activations
        std = (2.0 / (fan_in + fan_out)) ** 0.5 * scale ** 0.5
        with torch.no_grad():
            module.weight.normal_(0, std)
            if module.bias is not None:
                module.bias.zero_()

    elif isinstance(module, nn.Linear):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
        gain = calculate_gain('relu')  # Adjust gain for activation function
        std = (2.0 / (fan_in + fan_out)) ** 0.5 * scale ** 0.5
        with torch.no_grad():
            module.weight.normal_(0, std)
            if module.bias is not None:
                module.bias.zero_()

    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        with torch.no_grad():
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

# Apply initialization to the entire network
def initialize_network(network, scale=1.5):
    network.apply(lambda module: custom_weight_init(module, scale))
