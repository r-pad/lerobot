import torch
import torch.nn as nn
import torchvision.models as models


def make_depth_resnet18(pretrained=True, output_dim=512):
    """
    Create a ResNet18 encoder for depth (1 channel).
    Returns a nn.Module that outputs a feature vector of size `output_dim`.
    """
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )

    # convert conv1 from 3-channel → 1-channel
    old_weight = model.conv1.weight.data
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if pretrained:
        model.conv1.weight.data = old_weight.mean(dim=1, keepdim=True)

    # remove classification head
    modules = list(model.children())[:-1]
    encoder = nn.Sequential(*modules)  # output shape: (B,512,1,1)

    projection = nn.Linear(512, output_dim)
    return nn.Sequential(encoder, nn.Flatten(), projection)


class InsertionNet(nn.Module):
    """
    Implements Figure 5 Relation Architecture:
      Input1 → Encoder φ → f1
      Input2 → Encoder φ → f2
      concat → FCNN → action prediction
    """

    def __init__(self, feature_dim=512, action_dim=9, hidden_dim=512, shared_encoder=True):
        super().__init__()

        # If shared encoder: one φ is used for both inputs
        self.shared = shared_encoder
        self.encoder1 = make_depth_resnet18(pretrained=True, output_dim=feature_dim)

        if shared_encoder:
            self.encoder2 = self.encoder1
        else:
            self.encoder2 = make_depth_resnet18(pretrained=True, output_dim=feature_dim)

        # FCNN head (fully connected)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, depth1, depth2, action_gt=None):
        """
        depth1: (B,1,H,W) current observation
        depth2: (B,1,H,W) goal observation
        action_gt: (B, action_dim) ground truth relation action
        """
        f1 = self.encoder1(depth1)  # (B, feature_dim)
        f2 = self.encoder2(depth2)  # (B, feature_dim)

        fused = torch.cat([f1, f2], dim=-1)  # (B, 2*feature_dim)
        action_pred = self.mlp(fused)        # (B, action_dim)

        if action_gt is not None:
            loss = self.loss_fn(action_pred, action_gt)
            return loss

        return action_pred

if __name__ == "__main__":
    B = 4
    H, W = 480, 640
    action_dim = 9

    depth1 = torch.randn(B, 1, H, W)
    depth2 = torch.randn(B, 1, H, W)
    action_gt = torch.randn(B, action_dim)

    model = InsertionNet(
        feature_dim=512,
        hidden_dim=512,
        action_dim=action_dim
    )

    # Training mode → compute loss
    loss = model(depth1, depth2, action_gt)
    print("Training loss:", loss.item())

    # Inference mode → predicted action
    pred_action = model(depth1, depth2, action_gt=None)
    print("Pred action shape:", pred_action.shape)
    print("Pred actions:", pred_action)
