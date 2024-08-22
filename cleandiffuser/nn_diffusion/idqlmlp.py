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
            x_dim: int,
            emb_dim: int = 64,
            hidden_dim: int = 256,
            n_blocks: int = 3,
            dropout: float = 0.1,
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.Mish(), nn.Linear(emb_dim * 2, emb_dim))

        self.affine_in = nn.Linear(x_dim + emb_dim, hidden_dim)

        self.ln_resnet = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])

        self.affine_out = nn.Sequential(nn.Mish(), nn.Linear(hidden_dim, x_dim))

    def forward(self,
                x: torch.Tensor, t: torch.Tensor,
                condition: torch.Tensor = None):
        """
        Input:
            x:          (b, act_dim)
            noise:      (b, )
            condition:  (b, emb_dim)

        Output:
            y:          (b, act_dim)
        """
        emb = self.time_mlp(self.map_noise(t))
        if condition is not None:
           emb += condition
        x = torch.cat([x, emb], -1)
        x = self.affine_in(x)
        x = self.ln_resnet(x)
        return self.affine_out(x)
    