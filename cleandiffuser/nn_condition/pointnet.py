from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_condition import IdentityCondition


class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.max(x, -1)[0]


class SharedMlp(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_conv: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1) if use_conv else nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class Tnet(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        self.net = nn.Sequential(
            SharedMlp(k, 64),
            SharedMlp(64, 128),
            SharedMlp(128, 1024),
            MaxPool(),
            SharedMlp(1024, 512, use_conv=False),
            SharedMlp(512, 256, use_conv=False),
            nn.Linear(256, k * k),
        )
        self.iden = nn.Parameter(torch.eye(k).flatten().view(1, k * k))

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = x + self.iden
        return x.view(-1, self.k, self.k)


class FeatureTransform(nn.Module):
    """(..., 3, N) -> (..., 3, N)"""

    def __init__(self, in_dim: int):
        super().__init__()
        self.tnet = Tnet(in_dim)

    def forward(self, x: torch.Tensor):
        matrix = self.tnet(x)
        return torch.einsum("...ij,...jk->...ik", matrix, x)


class PointNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = FeatureTransform(3)
        self.shared_mlp1 = SharedMlp(3, 64)
        self.feature_transform = FeatureTransform(64)
        self.shared_mlp2 = nn.Sequential(SharedMlp(64, 128), SharedMlp(128, 1024), MaxPool())

    def forward(self, x: torch.Tensor):
        x = self.input_transform(x)
        x = self.shared_mlp1(x)
        local_feature = self.feature_transform(x)
        global_feature = self.shared_mlp2(local_feature)
        return local_feature, global_feature


class PointNetCondition(IdentityCondition):
    """PointNetCondition

    A PointNet used to extract features from point clouds.

    Args:
        emb_dim (int):
            The dimension of the output feature
        dropout (float):
            The label dropout rate, Default: 0.25

    Examples:
        >>> nn_condition = PointNetCondition(emb_dim=10)
        >>> x = torch.randn((2, 3, 64))  # (batch, 3, N)
        >>> nn_condition(x).shape
        torch.Size([2, 10])
    """

    def __init__(self, emb_dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = PointNetBackbone()
        self.post_process = nn.Sequential(
            SharedMlp(1024 + 64, 512, use_conv=False),
            SharedMlp(512, 256, use_conv=False),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        local_feature, global_feature = self.net(x)
        local_feature = torch.max(local_feature, -1)[0]
        feature = torch.cat([local_feature, global_feature], -1)
        return super().forward(self.post_process(feature), mask)


if __name__ == "__main__":
    x = torch.randn((2, 3, 1000))

    m = PointNetCondition(emb_dim=10)

    print(m(x).shape)
