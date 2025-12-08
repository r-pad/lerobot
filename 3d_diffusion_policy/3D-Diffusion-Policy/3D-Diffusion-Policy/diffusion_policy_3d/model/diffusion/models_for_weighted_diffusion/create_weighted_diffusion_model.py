from test_PointNet2.model_attn import AttnModel
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D

def create_weighted_diffusion_model(model_invariant, model_type, input_channels, device, input_dim=None, encoder_output_dim=None, separate_encoder_output_dim=None, load_pretrained_pointnet=None, diffusion_step_embed_dim=None, kernel_size=None, use_normalization=None, n_groups=None):
    if model_invariant:
            from test_PointNet2.model_invariant import PointNet2_small2
            from test_PointNet2.model_invariant import PointNet2
            from test_PointNet2.model_invariant import PointNet2_super
            from test_PointNet2.model_invariant import PointNet2_no_batch_norm
            if not use_normalization:
                if model_type == 'pointnet2':
                    print("!!!!!!!!!!!!!!!!!! USING MODEL INVARIANT POINT NET 2 NO NORMMMM !!!!!!")
                    model = PointNet2_no_batch_norm(num_classes=48).to(device)
            else:
                if model_type == 'pointnet2_small':
                    print("!!!!!!!!!!!!!!!!!! USING MODEL INVARIANT POINT NET 2 !!!!!!!")
                    model = PointNet2_small2(num_classes=48, input_channels=input_channels).to(device)
                elif model_type == 'pointnet2':
                    model = PointNet2(num_classes=48, input_channels=input_channels).to(device)
                elif model_type == 'pointnet2_super':
                    model = PointNet2_super(num_classes=48, input_channels=input_channels).to(device)
                elif model_type == 'attn':
                    model = AttnModel(num_classes=48).to(device)
                elif model_type == 'pointnet2_super_add_attn':
                    from test_PointNet2.model_invariant import PointNet2_super_add_attnention
                    model = PointNet2_super_add_attnention(num_classes=13, input_channels=input_channels).to(device)
                else:
                    raise ValueError(f"model_type {model_type} not recognized")
    else:
        from test_PointNet2.model import PointNet2_small2
        from test_PointNet2.model import PointNet2
        from test_PointNet2.model import PointNet2_super
        if model_type == 'pointnet2_small':
            model = PointNet2_small2(num_classes=48, input_channels=input_channels).to(device)
        elif model_type == 'pointnet2':
            model = PointNet2(num_classes=48, input_channels=input_channels).to(device)
        elif model_type == 'pointnet2_super':
            model = PointNet2_super(num_classes=48, input_channels=input_channels).to(device)
        elif model_type == 'attn':
            model = AttnModel(num_classes=48).to(device)
        elif model_type == 'unet':
            model = ConditionalUnet1D(
            input_dim=input_dim, # 12
            global_cond_dim=encoder_output_dim-1 + separate_encoder_output_dim if load_pretrained_pointnet else separate_encoder_output_dim, # for every point, use its own feature
            diffusion_step_embed_dim=diffusion_step_embed_dim, 
            down_dims=(128,128,128),
            kernel_size=kernel_size,
            n_groups=n_groups,
            use_group_norm=use_normalization,
        )
        else:
            raise ValueError(f"model_type {model_type} not recognized")
    return model
