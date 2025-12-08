from typing import Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops 
from einops.layers.torch import Rearrange
from termcolor import cprint
from diffusion_policy_3d.model.diffusor_actor_code.layers import FFWRelativeSelfCrossAttentionModule, FFWRelativeSelfAttentionModule, FFWRelativeCrossAttentionModule
from diffusion_policy_3d.model.vision.position_encodings import RotaryPositionEncoding3D
from diffusion_policy_3d.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy_3d.model.diffusion.conv1d_components import Conv1dBlock
from diffusion_policy_3d.model.diffusor_actor_code.current_gripper_encoder import Gripper_Encoder
from diffusion_policy_3d.model.diffusor_actor_code.utils import run_fps
logger = logging.getLogger(__name__)


class FilmConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=1
        ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
            Conv1dBlock(out_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
        ])

        cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.out_channels = out_channels
        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, cond=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        if cond is not None:
            embed = self.cond_encoder(cond)
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class FilmConditionalResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=1
        ):
        super().__init__()
        self.blocks = nn.ModuleList([
            FilmConditionalResidualBlock1D(in_channels,
                                             out_channels * 4,
                                             cond_dim,
                                             kernel_size,
                                             n_groups),
            FilmConditionalResidualBlock1D(out_channels * 4,
                                                out_channels * 4,
                                                cond_dim,
                                                kernel_size,
                                                n_groups),
            FilmConditionalResidualBlock1D(out_channels * 4,
                                                out_channels,
                                                cond_dim,
                                                kernel_size,
                                                n_groups),
        ])

    def forward(self, x, cond=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x, cond)
        out = self.blocks[1](out, cond)
        out = self.blocks[2](out, cond)
        return out



class ConditionalTransformer_3dda(nn.Module):
    def __init__(self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        scene_feature_dim=60,
        diffusion_attn_embed_dim=120,
        gripper_encoder_attn_head = 3,
        attention_module_num_attention_heads = 4,
        nhist=2,
        **kwargs
    ):
        cprint("========= USING CONDITIONAL TRANSFORMER =========", "green")
        if local_cond_dim is not None or global_cond_dim is None:
            cprint("Only support global condition for ConditionalTransformer now", "red")
            cprint("Contact Optimus Prime to update Transformers to support local condition", "red")
            raise NotImplementedError
        
        cprint("Only points action is supported for ConditionalTransformer now", "red")
        assert input_dim == 12, "Only points action is supported for ConditionalTransformer now"
        
        super().__init__()

        embedding_dim = diffusion_attn_embed_dim

        self.context_encoding = nn.Sequential(nn.Linear(scene_feature_dim, diffusion_attn_embed_dim), nn.ReLU()).cuda()

        self.relative_pe_emb = RotaryPositionEncoding3D(embedding_dim)

        self.gripper_encoder = Gripper_Encoder(embedding_dim=diffusion_attn_embed_dim, num_attn_heads=gripper_encoder_attn_head, nhist=2)

        self.traj_encoder = nn.Linear(3, embedding_dim) # Changed 9 to 12 from 3dda codebase

        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads = attention_module_num_attention_heads, num_layers=2, use_adaln=True
        )

        self.self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads = attention_module_num_attention_heads, num_layers=4, use_adaln=True
            )
        
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist * 4, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim) # 1*16*240
        ) # Changed from 3dda, in 3dda only embedding_dim * nhist. 

        self.position_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads = attention_module_num_attention_heads, num_layers = 2, use_adaln=True
            )


    
    def forward(self, 
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond: Union[torch.Tensor],
                observed_gripper_points: Union[torch.Tensor],
                scene_points: Union[torch.Tensor],
                scene_features: Union[torch.Tensor],
                num_gripper_points=4, 
                **kwargs):
        """
        Args:
            sample: (B, T, C) C = num_gripper_points * 3
            timestep: (B,)
            global_cond: (B, global_cond_dim)
            observed_gripper_points: (B, n_obs, num_gripper_points, 3)
            scene_points: (B, n_obs, N, 3)
            scene_features: (B, n_obs, N, encoder_feature_dim)
        """
        
        #import pdb; pdb.set_trace()
        B, T, _ = sample.shape
        n_obs = observed_gripper_points.shape[1]
        n_scene_points = scene_points.shape[2]
        timesteps = timestep
        timesteps = timesteps.expand(sample.shape[0]).cuda()
        sample_points = sample.reshape(B, T, num_gripper_points, 3) #(B, T, C) C = num_gripper_points * 3 to (B, T, num_gripper_points , 3)


        observed_gripper_points = einops.rearrange(
            observed_gripper_points, 'b n_obs N c -> b (n_obs N) c'
        )

        scene_points = einops.rearrange(
            scene_points, 'b n_obs N c -> b (n_obs N) c'
        )

        scene_features = einops.rearrange(
            scene_features, 'b n_obs N c -> b (n_obs N) c'
        )
        #import pdb; pdb.set_trace()
        scene_features = self.context_encoding(scene_features)
        current_gripper_features, _ = self.gripper_encoder.encode_curr_gripper(curr_gripper = observed_gripper_points, context_feats = scene_features, context = scene_points)
        #print("HEREEEEE",current_gripper_features.shape )
        #import pdb; pdb.set_trace()
        #print("current gripper features", current_gripper_features.shape, timesteps)
        time_embs = self.encode_denoising_timestep(timesteps, current_gripper_features)
        #import pdb; pdb.set_trace()
        sample_points_features = self.traj_encoder(sample_points.reshape(B, T * num_gripper_points, 3)) 
        #import pdb; pdb.set_trace()
        #rel_gripper_pos = self.relative_pe_layer(sample)
        #rel_context_pos = self.relative_pe_layer(scene_points)
        
        sample_points_pos_emb = self.relative_pe_emb(sample_points.reshape(B, T * num_gripper_points, 3)) # B, T * num_gripper_points, embedding_dim, 2
        #import pdb; pdb.set_trace() 
        '''scene_points = einops.rearrange(
            scene_points, 'b n_obs N c -> b (n_obs N) c'
        )'''
        scene_pos_emb = self.relative_pe_emb(scene_points)
        #import pdb; pdb.set_trace()
        '''sampled_context_features, sampled_rel_context_pos = run_fps(
            scene_features.transpose(0, 1),
            scene_pos_emb, fps_subsampling_factor=10
        )'''

        #import pdb; pdb.set_trace()
        #scene_features = self.context_encoding(scene_features)
        gripper_features = self.cross_attn(
            query=sample_points_features.transpose(0, 1),      # 1*16*240
            value=scene_features.transpose(0, 1),              # 1*9000*240
            query_pos=sample_points_pos_emb,   # 1*16*240*2
            value_pos=scene_pos_emb,           # 1*9000*240*2
            diff_ts=time_embs                  # 1*240
        )[-1]
        #import pdb; pdb.set_trace()
        sampled_context_features, sampled_rel_context_pos = run_fps(
            scene_features.transpose(0, 1),
            scene_pos_emb, fps_subsampling_factor=5
        )
        #import pdb; pdb.set_trace()
        # Self attention among gripper and sampled context
        features = torch.cat([gripper_features, sampled_context_features], 0) # scene_features.transpose(0, 1)
        rel_pos = torch.cat([sample_points_pos_emb, sampled_rel_context_pos], 1) # scene_pos_emb
        #import pdb; pdb.set_trace()
        features = self.self_attn(
            query=features,

            query_pos=rel_pos,
            diff_ts=time_embs
        )[-1]
        #import pdb; pdb.set_trace()
        num_gripper = sample_points_pos_emb.shape[1]
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper
        )
        position = position.reshape(B, T, num_gripper_points, 3)
        position = position.reshape(B, T, num_gripper_points * 3)
        #import pdb; pdb.set_trace()
        return position



    def predict_pos(self, features, rel_pos, time_embs, num_gripper):
        #import pdb; pdb.set_trace()
        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs
        )[-1]
        position_features = einops.rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features
    

    def encode_denoising_timestep(self, timestep, curr_gripper_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        #import pdb; pdb.set_trace()
        time_feats = self.time_emb(timestep)

        '''curr_gripper_features = einops.rearrange(
            curr_gripper_features, "npts b c -> b npts c"
        )'''
        curr_gripper_features = curr_gripper_features.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        return time_feats + curr_gripper_feats





        
        


        