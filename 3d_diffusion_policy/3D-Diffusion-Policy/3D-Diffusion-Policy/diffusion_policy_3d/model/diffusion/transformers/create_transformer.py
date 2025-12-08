from diffusion_policy_3d.model.diffusion.transformers.original_conditional_transformer import ConditionalTransformer
from diffusion_policy_3d.model.diffusion.transformers.self_attention_based_conditional_transformer import ConditionalTransformer_Self_Attention
from diffusion_policy_3d.model.diffusion.transformers.transformer_3dda import ConditionalTransformer_3dda


def create_conditional_transformer(transformer_type, local_cond_dim, input_dim, global_cond_dim, encoder_feature_dim, diffusion_attn_embed_dim, policy_type):
    if transformer_type == "default":
        print("Default Transformer is being used!!")
        return ConditionalTransformer(
                input_dim=input_dim,
                local_cond_dim=local_cond_dim,
                global_cond_dim=global_cond_dim,
                encoder_feature_dim=encoder_feature_dim,
                diffusion_attn_embed_dim=diffusion_attn_embed_dim,
                policy_type=policy_type,
            )
    elif transformer_type == "self_attention":
        print("Self Attention based Transformer is being used!!")
        return ConditionalTransformer_Self_Attention(
                input_dim=input_dim,
                local_cond_dim=local_cond_dim,
                global_cond_dim=global_cond_dim,
                encoder_feature_dim=encoder_feature_dim,
                diffusion_attn_embed_dim=diffusion_attn_embed_dim
            )
    elif transformer_type == "3dda":
        print("3D-Diffusion Actor based Transformer is being used!!")
        return ConditionalTransformer_3dda(
                input_dim=input_dim,
                local_cond_dim=local_cond_dim,
                global_cond_dim=global_cond_dim,
                encoder_feature_dim=encoder_feature_dim,
                diffusion_attn_embed_dim=diffusion_attn_embed_dim
            )