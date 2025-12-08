from diffusion_policy_3d.model.diffusor_actor_code.layers import FFWRelativeSelfCrossAttentionModule, FFWRelativeCrossAttentionModule
from diffusion_policy_3d.model.vision.position_encodings import RotaryPositionEncoding3D
import einops
import torch.nn as nn
class Gripper_Encoder():
    def __init__(self, embedding_dim, num_attn_heads, nhist=2):

        self.gripper_context_head = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=3, use_adaln=False
        ).cuda()
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim).cuda()
        self.curr_gripper_embed = nn.Embedding(4*nhist, embedding_dim).cuda()

        self.context_encoding = nn.Sequential(nn.Linear(60, 60), nn.ReLU()).cuda()

    def encode_curr_gripper(self, curr_gripper, context_feats, context):
        """
        Compute current gripper position features and positional embeddings.

        Args:
            - curr_gripper: (B, nhist, 3+)

        Returns:
            - curr_gripper_feats: (B, nhist, F)
            - curr_gripper_pos: (B, nhist, F, 2)
        """
        #import pdb; pdb.set_trace()
        #context_feats = self.context_encoding(context_feats)
        return self._encode_gripper(curr_gripper, self.curr_gripper_embed,
                                    context_feats, context)


    def _encode_gripper(self, gripper, gripper_embed, context_feats, context):
        """
        Compute gripper position features and positional embeddings.

        Args:
            - gripper: (B, npt, 3+)
            - context_feats: (B, npt, C)
            - context: (B, npt, 3)

        Returns:
            - gripper_feats: (B, npt, F)
            - gripper_pos: (B, npt, F, 2)
        """
        #import pdb; pdb.set_trace()
        # Learnable embedding for gripper
        #import pdb; pdb.set_trace()
        gripper_feats = gripper_embed.weight.unsqueeze(0).repeat(
            len(gripper), 1, 1
        )
        #gripper = gripper.reshape(1,1,-1)
        # Rotary positional encoding
        gripper_pos = self.relative_pe_layer(gripper[..., :3])
        context_pos = self.relative_pe_layer(context)

        gripper_feats = einops.rearrange(
            gripper_feats, 'b npt c -> npt b c'
        )
        context_feats = einops.rearrange(
            context_feats, 'b npt c -> npt b c'
        )
        gripper_feats = self.gripper_context_head(
            query=gripper_feats, value=context_feats,
            query_pos=gripper_pos, value_pos=context_pos
        )[-1]
        gripper_feats = einops.rearrange(
            gripper_feats, 'nhist b c -> b nhist c'
        )

        return gripper_feats, gripper_pos