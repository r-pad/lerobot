import torch
import torch.nn as nn
from monai.networks.nets import UNet


class FlowPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # Full-resolution U-Net (no downsampling)
        self.net = UNet(
            spatial_dims=2,
            in_channels=1,    # depth only
            out_channels=3,   # dense flow (dx, dy)
            channels=(32, 64, 128, 256, 512),
            strides=(1, 1, 1, 1),   # keep resolution = 480 × 640
        )

    def forward(self, cur_depth):
        """
        cur_depth: B * 1 * 480 * 640
        returns:   B * 2 * 480 * 640
        """
        return self.net(cur_depth)