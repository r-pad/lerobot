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
            out_channels=3,  # dense flow (dx, dy)
            channels=(32, 64, 128, 256, 512),
            strides=(1, 1, 1, 1),   # keep resolution = 480 × 640
            # norm=("GROUP", {"num_groups": 1})
            # norm = ("LAYER")
        )

    def forward(self, mask, cur_depth):
        """
        init_mask: B * 1 * 480 * 640
        cur_depth: B * 1 * 480 * 640
        returns:   B * 3 * 480 * 640
        """
        x = torch.cat([mask.unsqueeze(1), cur_depth], dim=1)
        return self.net(x)