from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.utils import GroupNorm1d

from .base_nn_diffusion import BaseNNDiffusion
from .jannerunet import Downsample1d, Upsample1d


class ChiResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, emb_dim: int, kernel_size: int = 3, cond_predict_scale: bool = False):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2),
            GroupNorm1d(out_dim, 8, 4), nn.Mish())
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size // 2),
            GroupNorm1d(out_dim, 8, 4), nn.Mish())

        cond_dim = 2 * out_dim if cond_predict_scale else out_dim
        self.cond_predict_scale = cond_predict_scale
        self.out_dim = out_dim
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(emb_dim, cond_dim))

        self.residual_conv = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, emb):
        out = self.conv1(x)
        embed = self.cond_encoder(emb)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_dim, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed.unsqueeze(-1)
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


class ChiUNet1d(BaseNNDiffusion):
    def __init__(
            self,
            act_dim: int,
            obs_dim: int,
            To: int,
            model_dim: int = 256,
            emb_dim: int = 256,
            kernel_size: int = 5,
            cond_predict_scale: bool = True,
            obs_as_global_cond: bool = True,
            dim_mult: List[int] = [1, 2, 2],
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.obs_as_global_cond = obs_as_global_cond
        self.model_dim = model_dim
        self.emb_dim = emb_dim

        dims = [act_dim] + [model_dim * m for m in np.cumprod(dim_mult)]

        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4), nn.Mish(),
            nn.Linear(emb_dim * 4, emb_dim))

        if obs_as_global_cond:
            self.global_cond_encoder = nn.Linear(To * obs_dim, emb_dim)
            emb_dim = emb_dim * 2  # cat obs and emb
            self.local_cond_encoder = None
        else:
            self.global_cond_encoder = None
            emb_dim = emb_dim
            self.local_cond_encoder = nn.ModuleList([
                ChiResidualBlock(
                    obs_dim, model_dim, emb_dim, kernel_size, cond_predict_scale),
                ChiResidualBlock(
                    obs_dim, model_dim, emb_dim, kernel_size, cond_predict_scale),
                Downsample1d(model_dim)])

        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ChiResidualBlock(
                    dim_in, dim_out, emb_dim, kernel_size, cond_predict_scale),
                ChiResidualBlock(
                    dim_out, dim_out, emb_dim, kernel_size, cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        self.mids = nn.ModuleList([
            ChiResidualBlock(
                mid_dim, mid_dim, emb_dim, kernel_size, cond_predict_scale),
            ChiResidualBlock(
                mid_dim, mid_dim, emb_dim, kernel_size, cond_predict_scale)])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ChiResidualBlock(
                    dim_out * 2, dim_in, emb_dim, kernel_size, cond_predict_scale),
                ChiResidualBlock(
                    dim_in, dim_in, emb_dim, kernel_size, cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size, padding=kernel_size // 2),
            GroupNorm1d(model_dim, 8, 4), nn.Mish(),
            nn.Conv1d(model_dim, act_dim, 1))

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, Ta, act_dim)
            noise:      (b, )
            condition:  (b, To, obs_dim)

        Output:
            y:          (b, Ta, act_dim)
        """
        # check Ta dimension
        assert x.shape[1] & (x.shape[1] - 1) == 0, "Ta dimension must be 2^n"

        x = x.permute(0, 2, 1)

        emb = self.map_noise(noise)
        emb = self.map_emb(emb)

        # If obs_as_global_cond, concatenate obs and emb
        if self.obs_as_global_cond:
            condition = self.global_cond_encoder(torch.flatten(condition, 1))
            if condition is not None:
                emb = torch.cat([emb, condition], dim=-1)
            else:
                emb = torch.cat([emb, torch.zeros_like(emb)], dim=-1)
            h_local = None
        else:
            condition = condition.permute(0, 2, 1)
            assert x.shape[-1] == condition.shape[-1]
            if condition is not None:
                resnet1, resnet2, dowmsample = self.local_cond_encoder
                h_local = [
                    resnet1(condition, emb), dowmsample(resnet2(condition, emb))]
            else:
                zero_cond = torch.zeros((x.shape[0], self.emb_dim, x.shape[1]))
                resnet1, resnet2, dowmsample = self.local_cond_encoder
                h_local = [
                    resnet1(condition, emb), dowmsample(resnet2(zero_cond, emb))]

        h = []

        for idx, (resnet1, resnet2, downsample) in enumerate(self.downs):
            x = resnet1(x, emb)
            if idx == 0 and h_local is not None:
                x = x + h_local[0]
            x = resnet2(x, emb)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mids:
            x = mid_module(x, emb)

        for idx, (resnet1, resnet2, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, emb)
            if idx == (len(self.ups) - 1) and h_local is not None:
                x = x + h_local[1]
            x = resnet2(x, emb)
            x = upsample(x)

        x = self.final_conv(x)

        x = x.permute(0, 2, 1)
        return x

