from typing import Optional

import einops
import torch
import torch.nn as nn

from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.utils import fps


class SharedMlp(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor):
        return self.net(x)


class MaxPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.max(self.dim)[0]


class DP3PointCloudCondition(IdentityCondition):
    """DP3 Point Cloud Condition.

    A lightweight PointNet from 3D Diffusion Policy (DP3) paper, https://arxiv.org/pdf/2403.03954.
    It maps the input point cloud to a fixed-size embedding.

    Args:
        emb_dim (int):
            The dimension of the output embedding.
        fps_downsample_points (Optional[int]):
            The number of points to downsample by FPS. If None, no downsampling is performed.

    Examples:
        >>> nn_condition = DP3PointCloudCondition(emb_dim=128, fps_downsample_points=512)
        >>> x = torch.randn((2, 3, 1000))  # (batch, 3, N)
        >>> nn_condition(x).shape
        torch.Size([2, 128])

    """

    def __init__(self, emb_dim: int, fps_downsample_points: Optional[int] = None):
        super().__init__()
        self.net = nn.Sequential(
            SharedMlp(3, 64),
            SharedMlp(64, 128),
            SharedMlp(128, 256),
            MaxPool(-2),
            nn.Linear(256, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.fps_downsample_points = fps_downsample_points

    def forward(self, condition: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.fps_downsample_points:
            _, condition = fps(condition, self.fps_downsample_points)
        condition = einops.rearrange(condition, "... d n -> ... n d")
        feature = self.net(condition)
        return super().forward(feature, mask)


if __name__ == "__main__":
    x = torch.randn((2, 3, 1000))

    m = DP3PointCloudCondition(emb_dim=128, fps_downsample_points=512)

    print(m(x).shape)
