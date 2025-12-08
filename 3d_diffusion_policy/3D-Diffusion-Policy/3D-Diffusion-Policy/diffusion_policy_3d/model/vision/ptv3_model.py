import torch
from torch import nn
from torchvision.ops import MLP

from ptv3.model import Point, PointTransformerV3


class LowlevelPTv3(nn.Module):
    def __init__(self, model_config, head_mlp_config, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.ptv3 = PointTransformerV3(**model_config)
        self.mlp_head = MLP(in_channels=self.ptv3.get_out_channels(), **head_mlp_config)

    def forward(self, x):
        B, N, C = x.shape
        # form data_dict
        offset = torch.arange(1, B + 1) * N
        data_dict = {
            "feat": x.reshape(-1, C),
            "coord": x[..., :3].reshape(-1, 3),
            "grid_size": self.grid_size,
            "offset": offset.to(x.device),
        }
        point = self.ptv3.forward(data_dict)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.mlp_head.forward(feat)
        seg_logits = seg_logits.reshape(B, N, -1)
        return seg_logits
