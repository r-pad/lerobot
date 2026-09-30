import torch
import torch.nn as nn
import torchvision.models as models

from policy.diffusion import DiffusionUNetPolicy


# -----------------------------
# Depth encoder: ResNet-18
# -----------------------------
def make_depth_resnet18(pretrained=True, output_dim=512):
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )

    old_weights = model.conv1.weight.data
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if pretrained:
        model.conv1.weight.data = old_weights.mean(dim=1, keepdim=True)

    modules = list(model.children())[:-1]
    encoder = nn.Sequential(*modules)

    proj = nn.Linear(512, output_dim)
    return nn.Sequential(encoder, nn.Flatten(), proj)


# -----------------------------
# Flow encoder: lightweight CNN
# -----------------------------
class FlowEncoder(nn.Module):
    """Encode (2, 240, 320) optical flow maps into a feature vector."""
    def __init__(self, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),  # 32×120×160
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 64×60×80
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),# 128×30×40
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 256×15×20
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, output_dim)

    def forward(self, flow):
        x = self.net(flow)  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.proj(x)    # (B, output_dim)
        return x


# -----------------------------
# Full Diffusion Policy
# -----------------------------
class Diffusion_Policy(nn.Module):
    def __init__(
        self,
        num_action=10,
        obs_feature_dim=512,
        flow_feature_dim=256,
        action_dim=9,
        proprio_dim=7,
        hidden_dim=512,
        pretrain=False
    ):
        super().__init__()

        # depth encoder (ResNet18)
        self.depth_encoder = make_depth_resnet18(
            pretrained=True, output_dim=obs_feature_dim
        )

        # FLOW encoder
        self.flow_encoder = FlowEncoder(output_dim=flow_feature_dim)

        # proprio MLP
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # fusion: depth + flow + proprio
        fusion_input_dim = obs_feature_dim + flow_feature_dim + hidden_dim
        self.fusion = nn.Linear(fusion_input_dim, hidden_dim)

        # diffusion action decoder
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, 1, hidden_dim)

        self.pretrain = pretrain

    def forward(self, flow=None, depth=None, proprioception=None, actions=None):
        """
        flow: (B, 2, 240, 320)
        depth: (B, 1, H, W)
        proprioception: (B, proprio_dim)
        """
        # encode depth
        depth_feat = self.depth_encoder(depth)     # (B, 512)

        # encode flow
        flow_feat = self.flow_encoder(flow)        # (B, 256)

        # encode proprio
        prop_feat = self.proprio_mlp(proprioception)  # (B, hidden_dim)

        # concat
        fused = torch.cat([depth_feat, flow_feat, prop_feat], dim=-1)
        readout = self.fusion(fused)

        # training
        if actions is not None:
            return self.action_decoder.compute_loss(readout, actions)

        # inference
        with torch.no_grad():
            return self.action_decoder.predict_action(readout)


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    B = 4
    H, W = 480, 640
    K = 20
    action_dim = 9

    depth = torch.randn(B, 1, H, W)
    flow = torch.randn(B, 2, 240, 320)
    proprio = torch.randn(B, 7)
    actions = torch.randn(B, K, action_dim)

    policy = Diffusion_Policy(num_action=K, action_dim=action_dim)

    loss = policy(depth=depth, flow=flow, proprioception=proprio, actions=actions)
    print("Training loss:", loss)

    pred_actions = policy(depth=depth, flow=flow, proprioception=proprio, actions=None)
    print("Pred actions:", pred_actions.shape)
