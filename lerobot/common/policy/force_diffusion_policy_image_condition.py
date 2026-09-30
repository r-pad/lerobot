import torch
import torch.nn as nn
import torchvision.models as models

from lerobot.common.policy.diffusion import DiffusionUNetPolicy

def make_depth_resnet18(pretrained=True, output_dim=512, in_channels=1):
    """
    ResNet18 encoder for depth-like input with configurable input channels.
    """
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )

    old_weights = model.conv1.weight.data  # (64, 3, 7, 7)
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    if pretrained:
        if in_channels == 3:
            model.conv1.weight.data = old_weights
        elif in_channels == 1:
            model.conv1.weight.data = old_weights.mean(dim=1, keepdim=True)
        else:
            avg_weight = old_weights.mean(dim=1, keepdim=True)  # (64,1,7,7)
            model.conv1.weight.data = avg_weight.repeat(1, in_channels, 1, 1) / in_channels

    modules = list(model.children())[:-1]  # keep until global avgpool
    encoder = nn.Sequential(*modules)
    proj = nn.Linear(512, output_dim)
    return nn.Sequential(encoder, nn.Flatten(), proj)

class Diffusion_Policy(nn.Module):
    def __init__(
        self,
        num_action=20,
        obs_feature_dim=512,
        action_dim=9,
        force_dim=3,
        hidden_dim=512,
        pretrain=False
    ):
        super().__init__()
        num_obs = 1

        # Original single-channel depth encoder
        self.depth_encoder = make_depth_resnet18(
            pretrained=True,
            output_dim=obs_feature_dim,
            in_channels=1
        )

        # 2-channel encoder for stacked plug/socket depth
        self.pair_depth_encoder = make_depth_resnet18(
            pretrained=True,
            output_dim=obs_feature_dim,
            in_channels=2
        )

        # Force MLP
        self.force_mlp = nn.Sequential(
            nn.Linear(force_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Fuse:
        # original depth feat + pair depth feat + force feat
        self.fusion = nn.Linear(
            obs_feature_dim + obs_feature_dim + 128,
            hidden_dim
        )

        # Action diffusion decoder
        self.action_decoder = DiffusionUNetPolicy(
            action_dim,
            num_action,
            num_obs,
            hidden_dim
        )

        self.pretrain = pretrain

    def forward(
        self,
        depth=None,                   # (B, 1, H, W) or (B, H, W)
        init_plug_photo_depth=None,   # (B, H, W)
        socket_depth=None,            # (B, H, W)
        force=None,                   # (B, force_dim)
        actions=None                  # (B, num_action, action_dim)
    ):
        """
        depth: original depth input
        init_plug_photo_depth: plug depth
        socket_depth: socket depth
        force: force input
        """

        # Make original depth shape (B, 1, H, W)
        if depth is None:
            raise ValueError("depth must be provided")

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)   # (B,H,W) -> (B,1,H,W)
        elif depth.ndim != 4:
            raise ValueError(
                f"depth must have shape (B,H,W) or (B,1,H,W), got {depth.shape}"
            )

        # Stack the two depth maps -> (B, 2, H, W)
        if init_plug_photo_depth is None or socket_depth is None:
            raise ValueError(
                "init_plug_photo_depth and socket_depth must both be provided"
            )

        pair_depth = torch.stack(
            [init_plug_photo_depth, socket_depth],
            dim=1
        )

        # Encode
        depth_feat = self.depth_encoder(depth)                 # (B, obs_feature_dim)
        pair_depth_feat = self.pair_depth_encoder(pair_depth)  # (B, obs_feature_dim)
        force_feat = self.force_mlp(force)                     # (B, 128)

        # Fuse
        fused = torch.cat(
            [depth_feat, pair_depth_feat, force_feat],
            dim=-1
        )
        readout = self.fusion(fused)

        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred

if __name__ == "__main__":
    B = 4
    H, W = 480, 640
    K = 20
    action_dim = 9

    depth = torch.randn(B, H, W)
    init_plug_photo_depth = torch.randn(B, H, W)
    socket_depth = torch.randn(B, H, W)
    proprio = torch.randn(B, 9)
    force = torch.randn(B, 3)
    actions = torch.randn(B, K, action_dim)

    policy = Diffusion_Policy(num_action=K, action_dim=action_dim)

    loss = policy(
        depth=depth,
        init_plug_photo_depth=init_plug_photo_depth,
        socket_depth=socket_depth,
        # proprioception=proprio,
        force=force,
        actions=actions
    )
    print("Training loss:", loss)

    pred_actions = policy(
        depth=depth,
        init_plug_photo_depth=init_plug_photo_depth,
        socket_depth=socket_depth,
        # proprioception=proprio,
        force=force,
        actions=None
    )
    print("Pred actions shape:", pred_actions.shape)

    
# import torch
# import torch.nn as nn
# import torchvision.models as models

# from policy.diffusion import DiffusionUNetPolicy


# def make_depth_resnet18(pretrained=True, output_dim=512, in_channels=1):
#     """
#     ResNet18 encoder for depth-like input with configurable input channels.
#     """
#     model = models.resnet18(
#         weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
#     )

#     old_weights = model.conv1.weight.data  # (64, 3, 7, 7)
#     model.conv1 = nn.Conv2d(
#         in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
#     )

#     if pretrained:
#         if in_channels == 3:
#             model.conv1.weight.data = old_weights
#         elif in_channels == 1:
#             model.conv1.weight.data = old_weights.mean(dim=1, keepdim=True)
#         else:
#             avg_weight = old_weights.mean(dim=1, keepdim=True)  # (64,1,7,7)
#             model.conv1.weight.data = avg_weight.repeat(1, in_channels, 1, 1) / in_channels

#     modules = list(model.children())[:-1]  # keep until global avgpool
#     encoder = nn.Sequential(*modules)
#     proj = nn.Linear(512, output_dim)
#     return nn.Sequential(encoder, nn.Flatten(), proj)


# class Diffusion_Policy(nn.Module):
#     def __init__(
#         self,
#         num_action=20,
#         obs_feature_dim=512,
#         action_dim=9,
#         proprio_dim=9,
#         force_dim=3,
#         hidden_dim=512,
#         pretrain=False
#     ):
#         super().__init__()
#         num_obs = 1

#         # Original single-channel depth encoder
#         self.depth_encoder = make_depth_resnet18(
#             pretrained=True,
#             output_dim=obs_feature_dim,
#             in_channels=1
#         )

#         # New 2-channel encoder for stacked plug/socket depth
#         self.pair_depth_encoder = make_depth_resnet18(
#             pretrained=True,
#             output_dim=obs_feature_dim,
#             in_channels=2
#         )

#         # Proprioception MLP
#         self.proprio_mlp = nn.Sequential(
#             nn.Linear(proprio_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )

#         # Force MLP
#         self.force_mlp = nn.Sequential(
#             nn.Linear(force_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128)
#         )

#         # Fuse:
#         # original depth feat + pair depth feat + proprio feat + force feat
#         self.fusion = nn.Linear(
#             obs_feature_dim + obs_feature_dim + hidden_dim + 128,
#             hidden_dim
#         )

#         # Action diffusion decoder
#         self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, hidden_dim)

#         self.pretrain = pretrain

#     def forward(
#         self,
#         depth=None,                   # (B, 1, H, W) or (B, H, W)
#         init_plug_photo_depth=None,   # (B, H, W)
#         socket_depth=None,            # (B, H, W)
#         proprioception=None,          # (B, proprio_dim)
#         force=None,                   # (B, force_dim)
#         actions=None                  # (B, num_action, action_dim)
#     ):
#         """
#         depth: original depth input
#         init_plug_photo_depth: new plug depth
#         socket_depth: new socket depth
#         """

#         # Make original depth shape (B,1,H,W)
#         if depth is None:
#             raise ValueError("depth must be provided")
#         if depth.ndim == 3:
#             depth = depth.unsqueeze(1)   # (B,H,W) -> (B,1,H,W)
#         elif depth.ndim != 4:
#             raise ValueError(f"depth must have shape (B,H,W) or (B,1,H,W), got {depth.shape}")

#         # Stack the two new depth maps -> (B,2,H,W)
#         if init_plug_photo_depth is None or socket_depth is None:
#             raise ValueError("init_plug_photo_depth and socket_depth must both be provided")
#         pair_depth = torch.stack([init_plug_photo_depth, socket_depth], dim=1)

#         # Encode
#         depth_feat = self.depth_encoder(depth)                 # (B, obs_feature_dim)
#         pair_depth_feat = self.pair_depth_encoder(pair_depth)  # (B, obs_feature_dim)
#         prop_feat = self.proprio_mlp(proprioception)           # (B, hidden_dim)
#         force_feat = self.force_mlp(force)                     # (B, 128)

#         # Fuse
#         fused = torch.cat([depth_feat, pair_depth_feat, prop_feat, force_feat], dim=-1)
#         readout = self.fusion(fused)

#         if actions is not None:
#             loss = self.action_decoder.compute_loss(readout, actions)
#             return loss
#         else:
#             with torch.no_grad():
#                 action_pred = self.action_decoder.predict_action(readout)
#             return action_pred


# if __name__ == "__main__":
#     B = 4
#     H, W = 480, 640
#     K = 20
#     action_dim = 9

#     depth = torch.randn(B, H, W)
#     init_plug_photo_depth = torch.randn(B, H, W)
#     socket_depth = torch.randn(B, H, W)
#     proprio = torch.randn(B, 9)
#     force = torch.randn(B, 3)
#     actions = torch.randn(B, K, action_dim)

#     policy = Diffusion_Policy(num_action=K, action_dim=action_dim)

#     loss = policy(
#         depth=depth,
#         init_plug_photo_depth=init_plug_photo_depth,
#         socket_depth=socket_depth,
#         proprioception=proprio,
#         force=force,
#         actions=actions
#     )
#     print("Training loss:", loss)

#     pred_actions = policy(
#         depth=depth,
#         init_plug_photo_depth=init_plug_photo_depth,
#         socket_depth=socket_depth,
#         proprioception=proprio,
#         force=force,
#         actions=None
#     )
#     print("Pred actions shape:", pred_actions.shape)