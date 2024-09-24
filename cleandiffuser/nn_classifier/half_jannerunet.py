from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.nn_diffusion.jannerunet import ResidualBlock, Downsample1d


class HalfJannerUNet1d(BaseNNDiffusion):
    """ Half JannerUNet1d for diffusion classifier. Adapted from https://github.com/jannerm/diffuser.

    Args:
        horizon: int,
            Length of the input sequence.
        in_dim: int,
            Number of input dimensions.
        out_dim: int,
            Number of output dimensions. Default is 1.
        kernel_size: int,
            Size of the convolution kernel. Default is 3.
        model_dim: int,
            Initial CNN model dimension. Default is 32.
        emb_dim: int,
            Number of dimensions in the embedding. Default is 32.
        dim_mult: Tuple[int],
            UNet dimension multiplier. Default is (1, 2, 2, 2).
        timestep_emb_type: str,
            Type of the timestep embedding. Default is "positional".
        norm_type: str,
            Type of the normalization layer. Default is "groupnorm".

    Examples:
        >>> nn_classifier = HalfJannerUNet1d(horizon=32, in_dim=10, out_dim=1, emb_dim=64)
        >>> x = torch.randn(2, 32, 10)
        >>> t = torch.randint(1000, (2,))
        >>> condition = torch.randn(2, 64)
        >>> nn_classifier(x, t).shape
        torch.Size([2, 1])
        >>> nn_classifier(x, t, condition).shape
        torch.Size([2, 1])
    """
    def __init__(
            self,
            horizon: int,
            in_dim: int,
            out_dim: int = 1,
            kernel_size: int = 3,
            model_dim: int = 32,
            emb_dim: int = 32,
            dim_mult: Tuple[int] = (1, 2, 2, 2),
            timestep_emb_type: str = "positional",
            norm_type: str = "groupnorm",
    ):
        super().__init__(emb_dim, timestep_emb_type)

        dims = [in_dim] + [model_dim * m for m in np.cumprod(dim_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, model_dim * 4), nn.Mish(),
            nn.Linear(model_dim * 4, model_dim))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualBlock(dim_in, dim_out, model_dim, kernel_size, norm_type),
                ResidualBlock(dim_out, dim_out, model_dim, kernel_size, norm_type),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = (horizon - 1) // 2 + 1

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4

        self.mid_block1 = nn.ModuleList([
            ResidualBlock(mid_dim, mid_dim_2, model_dim, kernel_size=5, norm_type=norm_type),
            Downsample1d(mid_dim_2)])
        horizon = (horizon - 1) // 2 + 1

        self.mid_block2 = nn.ModuleList([
            ResidualBlock(mid_dim_2, mid_dim_3, model_dim, kernel_size=5, norm_type=norm_type),
            Downsample1d(mid_dim_3)])
        horizon = (horizon - 1) // 2 + 1

        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + model_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim))

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):

        x = x.permute(0, 2, 1)

        emb = self.map_noise(noise)
        if condition is not None:
            emb = emb + condition
        emb = self.map_emb(emb)

        for resnet1, resnet2, downsample in self.downs:
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = downsample(x)

        x = self.mid_block1[0](x, emb)
        x = self.mid_block1[1](x)
        x = self.mid_block2[0](x, emb)
        x = self.mid_block2[1](x)

        x = x.flatten(1)
        out = self.final_block(torch.cat([x, emb], dim=-1))
        return out
