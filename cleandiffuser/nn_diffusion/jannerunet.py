from typing import List, Optional

import einops
import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import GroupNorm1d


def get_norm(dim: int, norm_type: str = "groupnorm"):
    if norm_type == "groupnorm":
        return GroupNorm1d(dim, 8, 4)
    elif norm_type == "layernorm":
        return LayerNorm(dim)
    else:
        return nn.Identity()


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, emb_dim: int, kernel_size: int = 3, norm_type: str = "groupnorm"):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2),
            get_norm(out_dim, norm_type), nn.Mish())
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size // 2),
            get_norm(out_dim, norm_type), nn.Mish())
        self.emb_mlp = nn.Sequential(
            nn.Mish(), nn.Linear(emb_dim, out_dim))
        self.residual_conv = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, emb):
        out = self.conv1(x) + self.emb_mlp(emb).unsqueeze(-1)
        out = self.conv2(out)
        return out + self.residual_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        out = self.to_out(out)
        return out + x


class JannerUNet1d(BaseNNDiffusion):
    def __init__(
            self,
            in_dim: int,
            model_dim: int = 32,
            emb_dim: int = 32,
            kernel_size: int = 3,
            dim_mult: List[int] = [1, 2, 2, 2],
            norm_type: str = "groupnorm",
            attention: bool = False,
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

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
                LinearAttention(dim_out) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, model_dim, kernel_size, norm_type)
        self.mid_attn = LinearAttention(mid_dim) if attention else nn.Identity()
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, model_dim, kernel_size, norm_type)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualBlock(dim_out * 2, dim_in, model_dim, kernel_size, norm_type),
                ResidualBlock(dim_in, dim_in, model_dim, kernel_size, norm_type),
                LinearAttention(dim_in) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 5, padding=2),
            get_norm(model_dim, norm_type), nn.Mish(),
            nn.Conv1d(model_dim, in_dim, 1))

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim) or None / No condition indicates zeros((b, emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        # check horizon dimension
        assert x.shape[1] & (x.shape[1] - 1) == 0, "Ta dimension must be 2^n"

        x = x.permute(0, 2, 1)

        emb = self.map_noise(noise)
        if condition is not None:
            emb = emb + condition
        else:
            emb = emb + torch.zeros_like(emb)
        emb = self.map_emb(emb)

        h = []

        for resnet1, resnet2, attn, downsample in self.downs:
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        for resnet1, resnet2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = x.permute(0, 2, 1)
        return x
