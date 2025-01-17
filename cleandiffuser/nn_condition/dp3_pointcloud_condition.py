from typing import List, Optional

import einops
import torch
import torch.nn as nn

from cleandiffuser.nn_condition import IdentityCondition


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
        >>> x = torch.randn((2, 1000, 3))  # (batch, N, 3)
        >>> nn_condition(x).shape
        torch.Size([2, 128])

    """

    def __init__(
        self,
        emb_dim: int,
        fps_downsample_points: Optional[int] = None,
        hidden_sizes: List[int] = (64, 128, 256),
    ):
        super().__init__()

        if fps_downsample_points is not None:
            try:
                from pytorch3d.ops import sample_farthest_points

                self.fps = sample_farthest_points
                self.torch3d_fps = True
            except ImportError:
                Warning("Pytorch3d not installed. Using slow fps implementation.")
                from cleandiffuser.utils import fps

                self.fps = fps
                self.torch3d_fps = False
        else:
            self.fps = None
            self.torch3d_fps = None

        layers = [SharedMlp(3, hidden_sizes[0])]
        for i, sz in enumerate(hidden_sizes[:-1]):
            layers.append(SharedMlp(sz, hidden_sizes[i + 1]))
        layers.extend([MaxPool(-2), nn.Linear(hidden_sizes[-1], emb_dim), nn.LayerNorm(emb_dim)])
        self.net = nn.Sequential(*layers)
        
        self.fps_downsample_points = fps_downsample_points

    def forward(self, condition: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.fps is not None:
            if self.torch3d_fps:
                condition = self.fps(condition, K=self.fps_downsample_points)[0]
            else:
                condition = self.fps(
                    einops.rearrange(condition, "... d n -> ... n d"), self.fps_downsample_points
                )[1]
        feature = self.net(condition)
        return super().forward(feature, mask)


if __name__ == "__main__":
    x = torch.randn((2, 1000, 3))

    m = DP3PointCloudCondition(emb_dim=128, fps_downsample_points=512)

    print(m(x).shape)
