from typing import List, Optional

import einops
import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion.base_nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import at_least_ndim


class GroupNorm2d(nn.Module):
    def __init__(self, dim, num_groups=32, eps=1e-6):
        super().__init__()
        assert dim % num_groups == 0, "dim must be divisible by num_groups"
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        emb_dim: int,
        kernel_size: int = 3,
        num_groups: int = 4,
        adaptive_scale: bool = True,
    ):
        super().__init__()
        self.adaptive_scale = adaptive_scale

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, padding=kernel_size // 2),
            GroupNorm2d(out_dim, num_groups),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size, padding=kernel_size // 2),
            GroupNorm2d(out_dim, num_groups),
            nn.Mish(),
        )
        self.emb_mlp = nn.Sequential(
            nn.Mish(), nn.Linear(emb_dim, out_dim * (2 if adaptive_scale else 1))
        )
        self.residual_conv = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, emb):
        if self.adaptive_scale:
            scale, shift = self.emb_mlp(emb).chunk(2, dim=-1)
            scale = at_least_ndim(scale, x.ndim)
            shift = at_least_ndim(shift, x.ndim)
        else:
            scale = 0.0
            shift = self.emb_mlp(emb)
            shift = at_least_ndim(shift, x.ndim)

        h = self.conv1(x)
        h = h * (1 + scale) + shift
        h = self.conv2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int = 4,
        attn_dropout: float = 0.0,
        bias: bool = True,
        num_groups: int = 4,
    ):
        super().__init__()
        assert dim % nhead == 0, "dim must be divisible by nhead"
        self.norm = GroupNorm2d(dim, num_groups)
        self.qkv_proj = nn.Conv2d(dim, dim * 3, 1)
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(
            dim, nhead, dropout=attn_dropout, bias=bias, batch_first=True
        )

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        qkv = self.qkv_proj(self.norm(x))
        qkv = einops.rearrange(qkv, "b c h w -> b (h w) c")
        q, k, v = qkv.chunk(qkv, dim=-1)
        out = self.attn(q, k, v)[0]
        out = einops.rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out + x


class UNet2d(BaseNNDiffusion):
    def __init__(
        self,
        n_channels: int,
        emb_dim: int = 32,
        model_dim: int = 32,
        kernel_size: int = 3,
        dim_mult: List[int] = [1, 2, 2],
        num_groups: int = 4,
        adaptive_scale: bool = True,
        num_blocks: int = 2,
        use_attention: bool = False,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        attn_bias: bool = True,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        dims = [n_channels] + [model_dim * m for m in np.cumprod(dim_mult)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.t_proj = nn.Sequential(
            nn.Linear(emb_dim, model_dim), nn.Mish(), nn.Linear(model_dim, model_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            res_blocks = nn.ModuleList(
                [
                    ResidualBlock(
                        dim_in, dim_out, model_dim, kernel_size, num_groups, adaptive_scale
                    ),
                    *[
                        ResidualBlock(
                            dim_out, dim_out, model_dim, kernel_size, num_groups, adaptive_scale
                        )
                        for _ in range(num_blocks - 1)
                    ],
                ]
            )

            self.downs.append(
                nn.ModuleList(
                    [
                        res_blocks,
                        AttentionBlock(dim_out, attn_heads, attn_dropout, attn_bias, num_groups)
                        if use_attention
                        else nn.Identity(),
                        nn.Conv2d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(
            mid_dim, mid_dim, model_dim, kernel_size, num_groups, adaptive_scale
        )
        self.mid_attn = (
            AttentionBlock(mid_dim, attn_heads, attn_dropout, attn_bias, num_groups)
            if use_attention
            else nn.Identity()
        )
        self.mid_block2 = ResidualBlock(
            mid_dim, mid_dim, model_dim, kernel_size, num_groups, adaptive_scale
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            res_blocks = nn.ModuleList(
                [
                    ResidualBlock(
                        dim_out * 2, dim_in, model_dim, kernel_size, num_groups, adaptive_scale
                    ),
                    *[
                        ResidualBlock(
                            dim_in, dim_in, model_dim, kernel_size, num_groups, adaptive_scale
                        )
                        for _ in range(num_blocks - 1)
                    ],
                ]
            )

            self.ups.append(
                nn.ModuleList(
                    [
                        res_blocks,
                        AttentionBlock(dim_in, attn_heads, attn_dropout, attn_bias, num_groups)
                        if use_attention
                        else nn.Identity(),
                        nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, 3, padding=1),
            GroupNorm2d(model_dim, num_groups),
            nn.Mish(),
            nn.Conv2d(model_dim, n_channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None):
        emb = self.t_proj(self.map_noise(t))

        if condition is not None:
            emb = emb + condition

        h = []

        for resnets, attn, downsample in self.downs:
            for resnet in resnets:
                x = resnet(x, emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        for resnets, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            for resnet in resnets:
                x = resnet(x, emb)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    model = UNet2d(n_channels=3, model_dim=32, num_blocks=2, num_groups=4, adaptive_scale=True)
    x = torch.randn((2, 3, 20, 20))
    t = torch.randint(0, 1000, (2,))
    print(model(x, t).shape)
