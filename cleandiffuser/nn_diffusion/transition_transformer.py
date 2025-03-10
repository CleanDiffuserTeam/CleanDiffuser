from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion.base_nn_diffusion import BaseNNDiffusion

__all__ = ["TransitionTransformer"]


class TransitionTransformer(BaseNNDiffusion):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        emb_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        timestep_emb_type: str = "untrainable_fourier",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(d_model, timestep_emb_type, timestep_emb_params)
        self.obs_dim, self.act_dim, self.emb_dim = obs_dim, act_dim, emb_dim

        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.act_proj = nn.Linear(act_dim, d_model)
        self.rew_proj = nn.Linear(1, d_model)
        self.tml_proj = nn.Linear(1, d_model)

        self.obs_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.next_obs_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.act_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.rew_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.tml_emb = nn.Parameter(torch.randn(1, 1, d_model))

        self.tfm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, batch_first=True), num_layers
        )

        self.obs_out = nn.Linear(d_model, obs_dim)
        self.act_out = nn.Linear(d_model, act_dim)
        self.rew_out = nn.Linear(d_model, 1)
        self.tml_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None):
        obs = x[:, : self.obs_dim]
        next_obs = x[:, self.obs_dim : self.obs_dim * 2]
        rew = x[:, self.obs_dim * 2 : self.obs_dim * 2 + 1]
        act = x[:, self.obs_dim * 2 + 1 : self.obs_dim * 2 + 1 + self.act_dim]
        tml = x[:, self.obs_dim * 2 + 1 + self.act_dim :]

        cond = self.map_noise(t).unsqueeze(1)
        if condition is not None:
            cond += condition

        obs = self.obs_proj(obs)[:, None] + self.obs_emb
        next_obs = self.obs_proj(next_obs)[:, None] + self.next_obs_emb
        act = self.act_proj(act)[:, None] + self.act_emb
        rew = self.rew_proj(rew)[:, None] + self.rew_emb
        tml = self.tml_proj(tml)[:, None] + self.tml_emb

        x = torch.cat([obs, next_obs, rew, act, tml, cond], 1)
        x = self.tfm(x)
        obs = self.obs_out(x[:, 0])
        next_obs = self.obs_out(x[:, 1])
        rew = self.rew_out(x[:, 2])
        act = self.act_out(x[:, 3])
        tml = self.tml_out(x[:, 4])

        return torch.cat([obs, next_obs, rew, act, tml], -1)
