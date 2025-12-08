from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from termcolor import cprint
import copy
import time
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet2_utils import PointNet2_small2


class WeightedDisplacementPolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler,
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
            **kwargs):
        super().__init__()

        self.model_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PointNet2_small2(num_classes=13).to(self.model_device)
        assert prediction_target == 'goal_gripper_pcd', f"prediction_target {prediction_target} not recognized"
        self.prediction_target = prediction_target

        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        self.normalizer = LinearNormalizer()


    def predict_action(self, obs_dict):
        nobs = obs_dict
        pointcloud = nobs['point_cloud'][:, self.n_obs_steps-1, :, :]
        gripper_pcd = nobs['gripper_pcd'][:, self.n_obs_steps-1, :, :]
        import pdb; pdb.set_trace()

        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
        B, N, _ = inputs.shape
        inputs = inputs.to(self.model_device)
        inputs = inputs.permute(0, 2, 1)
        outputs = self.model(inputs)
        weights = outputs[:, :, -1]
        outputs = outputs[:, :, :-1]

        inputs = inputs.permute(0, 2, 1)
        outputs = outputs.view(B, N, 4, 3)
        outputs = outputs + inputs.unsqueeze(2) # B, N, 4, 3

        # softmax the weights
        weights = torch.nn.functional.softmax(weights, dim=1)

        # sum the displacement of the predicted gripper point cloud according to the weights
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)
        import pdb; pdb.set_trace()

        outputs = outputs.unsqueeze(1).repeat(1, self.n_action_steps, 1, 1)
        outputs = outputs.view(B, -1, 12)
        import pdb; pdb.set_trace()
        result = {
            'action': outputs,
            'action_pred': outputs,
        }

        return result

        


    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(copy.deepcopy(normalizer.state_dict()))

    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_loss(self, batch):
        import pdb; pdb.set_trace()
        criterion = torch.nn.MSELoss()
        nobs = batch['obs']

        nactions = batch['obs'][self.prediction_target]

        goal_gripper_pcd = nactions[:, self.n_obs_steps-1, :, :]

        batch_size = nactions.shape[0]

        gripper_points = goal_gripper_pcd
        pointcloud = nobs['point_cloud'][:, self.n_obs_steps-1, :, :]
        gripper_pcd = nobs['gripper_pcd'][:, self.n_obs_steps-1, :, :]

        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
        labels = gripper_points.unsqueeze(1) - inputs.unsqueeze(2)
        B, N, _, _ = labels.shape
        labels = labels.view(B, N, -1)

        inputs, labels = inputs.to(self.model_device), labels.to(self.model_device)
        inputs = inputs.permute(0, 2, 1)
        outputs = self.model(inputs)
        weights = outputs[:, :, -1]
        outputs = outputs[:, :, :-1]

        loss = criterion(outputs, labels)

        inputs = inputs.permute(0, 2, 1)
        outputs = outputs.view(B, N, 4, 3)
        outputs = outputs + inputs.unsqueeze(2) # B, N, 4, 3

        # softmax the weights
        weights = torch.nn.functional.softmax(weights, dim=1)
        
        # sum the displacement of the predicted gripper point cloud according to the weights
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)
        avg_loss = criterion(outputs, gripper_points.to(self.model_device))

        loss = loss + avg_loss * 10

        loss_dict = {
            'bc_loss': loss.item(),
        }
        return loss, loss_dict


