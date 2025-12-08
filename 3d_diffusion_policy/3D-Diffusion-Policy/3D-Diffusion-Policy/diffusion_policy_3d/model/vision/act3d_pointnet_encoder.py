import torch
import torch.nn as nn
from diffusion_policy_3d.model.vision.layers import RelativeCrossAttentionModule
from diffusion_policy_3d.common.network_helper import replace_bn_with_gn
from diffusion_policy_3d.model.vision.position_encodings import RotaryPositionEncoding3D
from diffusion_policy_3d.model.vision.pointnet_extractor import create_mlp
# import segmentation_models_pytorch as smp
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import einops
from diffusion_policy_3d.model.vision.pointnet2_utils import PointNet2, SimpleMLP

class Act3dPointNetEncoder(nn.Module):
    def __init__(self, 
                 encoder_output_dim=256, 
                 num_gripper_points=4, 
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 observation_space=None,
                 encoder_type='act3d_pointnet',
                 **kwargs
                 ):
        super(Act3dPointNetEncoder, self).__init__()
        
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.gripper_pcd_key = 'gripper_pcd'
        self.num_gripper_points = num_gripper_points
        self.encoder_output_dim = encoder_output_dim
        self.state_shape = observation_space[self.state_key]
        
        if encoder_type == 'act3d_pointnet':
            vision_encoder = PointNet2(num_classes=encoder_output_dim)
            vision_encoder = replace_bn_with_gn(vision_encoder, features_per_group=4)
        elif encoder_type == 'act3d_mlp':
            vision_encoder = SimpleMLP(num_classes=encoder_output_dim)

        attn_layers = RelativeCrossAttentionModule(encoder_output_dim, 4, 2)
        attn_layers = replace_bn_with_gn(attn_layers)
        
        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'relative_pe_layer': RotaryPositionEncoding3D(encoder_output_dim),
            'embed': nn.Embedding(1, encoder_output_dim),
            'attn_layers': attn_layers,
        })
        
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels = encoder_output_dim * self.num_gripper_points
        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observation: Dict) -> torch.Tensor:
        # TODO: the things passed in is already flattend from B, T, ... -> B*T, ...
        
        nets = self.nets
        
        # TODO: check the input shape
        agent_pos = observation[self.state_key]
        B = agent_pos.shape[0] #  B = batch_size * obs_horizon

        # NOTE: use PointNet++ to extract features from the point cloud
        pc_obs = observation[self.point_cloud_key]
        B, N, c = pc_obs.shape
        rgb_obs = einops.rearrange(pc_obs, "B N c -> B c N") # NOTE: input to pointnet is B 3 N
        rgb_features, _ = nets['vision_encoder'](rgb_obs) # shape B N c
        rgb_features = einops.rearrange(rgb_features, "B N c -> N B c") # shape N B c
        
            
        point_cloud = observation[self.point_cloud_key]
        # TODO: this can be done when retrieving the point cloud
        # point_cloud = einops.rearrange(point_cloud, "B c h w -> B (h w) c", B=B) # NOTE: our pcd comes in as B N 3, where N = h*w is the image size
        point_cloud_rel_pos_embedding = nets['relative_pe_layer'](point_cloud) # shape B N encoder_output_dim
                       
        num_gripper_points = observation['gripper_pcd'].shape[1] # gripper pcd is B num_gripper_points 3
        assert num_gripper_points == self.num_gripper_points, f"Expected {self.num_gripper_points} gripper points, got {num_gripper_points}"
        gripper_pcd = observation[self.gripper_pcd_key]
        gripper_pcd_rel_pos_embedding = nets['relative_pe_layer'](gripper_pcd) # shape B num_gripper_points encoder_output_dim
        gripper_pcd_features = nets['embed'].weight.unsqueeze(0).repeat(num_gripper_points, B, 1) # shape (num_gripper_points, B, encoder_output_dim)

        # TODO: we can further modify this such that the gripper only attends to the object point cloud
        attn_output = nets['attn_layers'](
            query=gripper_pcd_features, value=rgb_features,
            query_pos=gripper_pcd_rel_pos_embedding, value_pos=point_cloud_rel_pos_embedding,
        )[-1]
        
        rgb_features = einops.rearrange(
            attn_output, "num_gripper_points B embed_dim -> B num_gripper_points embed_dim").flatten(start_dim=1) # shape B (num_gripper_points * encoder_output_dim)

        state_feat = self.state_mlp(agent_pos)  # B * 64
        # print('rgb_features ', rgb_features.shape)
        # print('agent_pos ', state_feat.shape)
        
        obs_features = torch.cat([rgb_features, state_feat], dim=-1)
        return obs_features
    
    def output_shape(self):
        return self.n_output_channels

        