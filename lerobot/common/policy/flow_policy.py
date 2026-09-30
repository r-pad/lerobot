import torch
import torch.nn as nn
from monai.networks.nets import UNet

class FlowPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # Full-resolution U-Net (no downsampling)
        self.net = UNet(
            spatial_dims=2,
            in_channels=2,   # mask + depth
            out_channels=2,  # dense flow (dx, dy)
            channels=(32, 64, 128, 256, 512),
            strides=(1, 1, 1, 1),   # keep resolution = 480 × 640
            norm=("GROUP", {"num_groups":8})
        )

    def forward(self, init_mask, cur_depth):
        """
        init_mask: B * 1 * 480 * 640
        cur_depth: B * 1 * 480 * 640
        returns:   B * 2 * 480 * 640
        """
        x = torch.cat([init_mask, cur_depth], dim=1)
        return self.net(x)
def test_flow_policy():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = FlowPolicy().to(device)
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            print(name, "->", module)
    B = 2
    H, W = 240 , 320

    init_mask = torch.rand(B, 1, H, W, device=device)
    cur_depth = torch.rand(B, 1, H, W, device=device)

    with torch.no_grad():
        flow = model(init_mask, cur_depth)

    print("Output shape:", flow.shape)
    assert flow.shape == (B, 2, H, W), "Shape mismatch!"
    print("✅ Passed: Output is B*2*480*640")

if __name__ == "__main__":
    test_flow_policy()
    