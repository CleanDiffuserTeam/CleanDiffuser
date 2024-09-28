from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4), nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, x):
        return x + self.net(x)


class IDQLMlp(BaseNNDiffusion):
    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            emb_dim: int = 64,
            hidden_dim: int = 256,
            n_blocks: int = 3,
            dropout: float = 0.1,
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.obs_dim = obs_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.Mish(), nn.Linear(emb_dim * 2, emb_dim))

        self.affine_in = nn.Linear(obs_dim + act_dim + emb_dim, hidden_dim)

        self.ln_resnet = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])

        self.affine_out = nn.Linear(hidden_dim, act_dim)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, act_dim)
            noise:      (b, )
            condition:  (b, obs_dim)

        Output:
            y:          (b, act_dim)
        """
        if condition is None:
            condition = torch.zeros(x.shape[0], self.obs_dim).to(x.device)

        t = self.time_mlp(self.map_noise(noise))
        x = torch.cat([x, t, condition], -1)
        x = self.affine_in(x)
        x = self.ln_resnet(x)

        return self.affine_out(x)


class NewIDQLMlp(BaseNNDiffusion):
    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            emb_dim: int = 64,
            hidden_dim: int = 256,
            n_blocks: int = 3,
            dropout: float = 0.1,
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.obs_dim = obs_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.Mish(), nn.Linear(emb_dim * 2, emb_dim))

        self.affine_in = nn.Linear(obs_dim + act_dim + emb_dim, hidden_dim)

        self.ln_resnet = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])

        self.affine_out = nn.Sequential(nn.Mish(), nn.Linear(hidden_dim, act_dim))

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, act_dim)
            noise:      (b, )
            condition:  (b, obs_dim)

        Output:
            y:          (b, act_dim)
        """
        if condition is None:
            condition = torch.zeros(x.shape[0], self.obs_dim).to(x.device)

        t = self.time_mlp(self.map_noise(noise))
        x = torch.cat([x, t, condition], -1)
        x = self.affine_in(x)
        x = self.ln_resnet(x)

        return self.affine_out(x)
    