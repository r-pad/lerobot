import torch
import torch.nn as nn
import torchvision.models as models

from policy.diffusion import DiffusionUNetPolicy


def make_rgb_resnet18(pretrained=True, output_dim=512):
    """
    RGB encoder based on ResNet18.
    If pretrained=True, load ImageNet weights.
    """
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )
    # conv1 already expects 3-channel RGB
    modules = list(model.children())[:-1]  # keep until global avgpool
    encoder = nn.Sequential(*modules)
    proj = nn.Linear(512, output_dim)
    return nn.Sequential(encoder, nn.Flatten(), proj)


class Diffusion_Policy(nn.Module):
    def __init__(
        self, 
        num_action=10,
        obs_feature_dim=512, 
        action_dim=9,     # (3 delta pos + 6 rot6d)
        proprio_dim=9,    # (3 pos + 6 rot6d)
        hidden_dim=512,
        pretrain=False
    ):
        super().__init__()
        num_obs = 1

        # rgb encoder (ResNet18 backbone)
        self.rgb_encoder = make_rgb_resnet18(pretrained=True, output_dim=obs_feature_dim)

        # proprioception MLP
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # fusion layer (rgb + proprio → readout)
        self.fusion = nn.Linear(obs_feature_dim + hidden_dim, hidden_dim)

        # action diffusion decoder
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, hidden_dim)

        self.pretrain = pretrain

    def forward(self, rgb=None, proprioception=None, actions=None):
        """
        rgb: (B, H, W, 3) or (B, 3, H, W), dtype uint8 in [0,255]
        proprioception: (B, proprio_dim)
        actions: (B, num_action, action_dim) if training
        """
        # ensure channel-first
        if rgb.ndim == 4 and rgb.shape[-1] == 3:   # (B, H, W, 3)
            rgb = rgb.permute(0, 3, 1, 2)          # -> (B, 3, H, W)

        # convert to float and normalize to [0,1]
        rgb = rgb.float() / 255.0  

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb.device)[None, :, None, None]
        rgb = (rgb - mean) / std

        # encode rgb
        rgb_feat = self.rgb_encoder(rgb)  # (B, obs_feature_dim)

        # encode proprioception
        prop_feat = self.proprio_mlp(proprioception)  # (B, hidden_dim)

        # fuse
        fused = torch.cat([rgb_feat, prop_feat], dim=-1)  # (B, obs_feature_dim + hidden_dim)
        readout = self.fusion(fused)  # (B, hidden_dim)
        
        if actions is not None:  # training mode
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:  # inference mode
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred

if __name__ == "__main__":
    # fake batch
    B = 4
    H, W = 480, 640
    K = 20   # number of action steps
    action_dim = 9

    # Simulated RGB frames: (B, 3, H, W)
    rgb = torch.randint(0, 256, (B, 3, H, W), dtype=torch.uint8)

    # Simulated proprioception: (B, 9)
    proprio = torch.randn(B, 9)

    # Simulated future actions: (B, K, action_dim)
    actions = torch.randn(B, K, action_dim)

    # build policy
    policy = Diffusion_Policy(num_action=K, action_dim=action_dim)

    # training mode
    loss = policy(rgb=rgb, proprioception=proprio, actions=actions)
    print("Training loss:", loss.item())

    # inference mode
    pred_actions = policy(rgb=rgb, proprioception=proprio, actions=None)
    print("Pred actions shape:", pred_actions.shape)  # expect (B, K, action_dim)