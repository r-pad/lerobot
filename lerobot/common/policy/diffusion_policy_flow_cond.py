import torch
import torch.nn as nn
import torchvision.models as models

from policy.diffusion import DiffusionUNetPolicy


# ------------------------------------------------------------
# Encoder: Depth (1) + Flow (2) -> 3-channel ResNet18
# ------------------------------------------------------------
def make_depth_flow_resnet18(pretrained=True, output_dim=512):
    """
    Encoder for stacked [depth, flow_x, flow_y] input (3-channel).
    Uses standard ResNet18 ImageNet weights.
    """
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )

    # Remove classification head, keep backbone + global avgpool
    modules = list(model.children())[:-1]  # up to avgpool
    encoder = nn.Sequential(*modules)

    proj = nn.Linear(512, output_dim)

    return nn.Sequential(
        encoder,
        nn.Flatten(),
        proj
    )


# ------------------------------------------------------------
# Diffusion Policy
# ------------------------------------------------------------
class Diffusion_Policy(nn.Module):
    def __init__(
        self,
        num_action=20,
        obs_feature_dim=512,
        action_dim=9,      # (3 delta pos + 6 rot6d)
        proprio_dim=7,     # proprio input dim
        hidden_dim=512,
        pretrain=False
    ):
        super().__init__()
        num_obs = 1

        # Depth + Flow encoder
        self.depth_flow_encoder = make_depth_flow_resnet18(
            pretrained=True,
            output_dim=obs_feature_dim
        )

        # Proprioception MLP
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Fusion layer
        self.fusion = nn.Linear(obs_feature_dim + hidden_dim, hidden_dim)

        # Diffusion action decoder
        self.action_decoder = DiffusionUNetPolicy(action_dim,num_action,num_obs,hidden_dim)

        self.pretrain = pretrain

    def forward(self, depth=None, proprioception=None, actions=None, flow = None):
        """
        Args:
            depth: (B, 1, H, W)
            proprioception: (B, proprio_dim)
            actions: (B, num_action, action_dim) if training

        Returns:
            loss (training) or predicted actions (inference)
        """
        assert depth is not None
        assert proprioception is not None

        B, _, H, W = depth.shape

        # ----------------------------------------------------
        # Debug step: initialize flow as zeros
        # ----------------------------------------------------
        #flow = torch.zeros(
        #    B, 2, H, W,
        #    device=depth.device,
        #    dtype=depth.dtype
        #)

        # Stack depth + flow -> 3 channels
        depth_flow = torch.cat([depth, flow], dim=1)  # (B, 3, H, W)

        # Encode visual observation
        obs_feat = self.depth_flow_encoder(depth_flow)  # (B, obs_feature_dim)

        # Encode proprioception
        prop_feat = self.proprio_mlp(proprioception)  # (B, hidden_dim)

        # Fuse
        fused = torch.cat([obs_feat, prop_feat], dim=-1)
        readout = self.fusion(fused)  # (B, hidden_dim)

        # Diffusion policy
        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred


# ------------------------------------------------------------
# Simple sanity check
# ------------------------------------------------------------
if __name__ == "__main__":
    B = 4
    H, W = 480, 640
    K = 20
    action_dim = 9
    proprio_dim = 7

    depth = torch.randn(B, 1, H, W)
    proprio = torch.randn(B, proprio_dim)
    actions = torch.randn(B, K, action_dim)

    policy = Diffusion_Policy(
        num_action=K,
        action_dim=action_dim,
        proprio_dim=proprio_dim
    )

    # Training mode
    loss = policy(depth=depth, proprioception=proprio, actions=actions)
    print("Training loss:", loss.item())

    # Inference mode
    pred_actions = policy(depth=depth, proprioception=proprio)
    print("Pred actions shape:", pred_actions.shape)