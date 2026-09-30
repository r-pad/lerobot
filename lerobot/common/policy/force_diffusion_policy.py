import torch
import torch.nn as nn
import torchvision.models as models

from policy.diffusion import DiffusionUNetPolicy


def make_depth_resnet18(pretrained=True, output_dim=512):
    """
    Depth encoder based on ResNet18, adapted for 1-channel input.
    If pretrained=True, load ImageNet weights and average conv1 weights to 1 channel.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    # change conv1 to 1-channel
    old_weights = model.conv1.weight.data
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        # average RGB weights → depth channel
        model.conv1.weight.data = old_weights.mean(dim=1, keepdim=True)
    # remove the classifier head, keep encoder
    modules = list(model.children())[:-1]  # keep until global avgpool
    encoder = nn.Sequential(*modules)
    proj = nn.Linear(512, output_dim)
    return nn.Sequential(encoder, nn.Flatten(), proj)

class Diffusion_Policy(nn.Module):
    def __init__(
        self, 
        num_action=20,
        obs_feature_dim=512, 
        action_dim=9,     # (3 delta pos + 6 rot6d)
        proprio_dim=9,    # (3 pos + 6 rot6d)
        force_dim=3,      # Added: 3D Force input
        hidden_dim=512,
        pretrain=False
    ):
        super().__init__()
        num_obs = 1

        # 1. Depth encoder (ResNet18 backbone)
        self.depth_encoder = make_depth_resnet18(pretrained=True, output_dim=obs_feature_dim)

        # 2. Proprioception MLP
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 3. Force MLP (Added)
        self.force_mlp = nn.Sequential(
            nn.Linear(force_dim, 128), # Force is lower dimensional, smaller MLP
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # 4. Updated Fusion layer (depth + proprio + force → readout)
        # Input dim: obs_feature_dim (512) + proprio_hidden (512) + force_hidden (128)
        self.fusion = nn.Linear(obs_feature_dim + hidden_dim + 128, hidden_dim)

        # 5. Action diffusion decoder
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, hidden_dim)

        self.pretrain = pretrain

    def forward(self, depth=None, proprioception=None, force=None, actions=None):
        """
        depth: (B, 1, H, W)
        proprioception: (B, proprio_dim)
        force: (B, 3)
        actions: (B, num_action, action_dim) if training
        """
        # Encode modalities
        depth_feat = self.depth_encoder(depth)           # (B, obs_feature_dim)
        prop_feat = self.proprio_mlp(proprioception)     # (B, hidden_dim)
        force_feat = self.force_mlp(force)               # (B, 128)

        # Fuse all three modalities
        fused = torch.cat([depth_feat, prop_feat, force_feat], dim=-1) 
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
    K = 20
    action_dim = 9

    depth = torch.randn(B, 1, H, W)
    proprio = torch.randn(B, 9)
    force = torch.randn(B, 3)          # New force vector
    actions = torch.randn(B, K, action_dim)

    # build policy
    policy = Diffusion_Policy(num_action=K, action_dim=action_dim)

    # training mode test
    loss = policy(depth=depth, proprioception=proprio, force=force, actions=actions)
    print("Training loss:", loss)

    # inference mode test
    pred_actions = policy(depth=depth, proprioception=proprio, force=force, actions=None)
    print("Pred actions shape:", pred_actions.shape)