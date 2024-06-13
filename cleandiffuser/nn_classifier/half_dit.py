from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion.dit import DiT1d, FinalLayer1d


class HalfDiT1d(DiT1d):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 emb_dim: int,
                 d_model: int = 384,
                 n_heads: int = 6,
                 depth: int = 12,
                 dropout: float = 0.0,
                 timestep_emb_type: str = "positional",
                 ):
        super().__init__(in_dim, emb_dim, d_model, n_heads, depth, dropout, timestep_emb_type)
        self.final_layer = FinalLayer1d(d_model, d_model // 2)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        self.proj = nn.Sequential(
            nn.LayerNorm(d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4), nn.SiLU(),
            nn.Linear(d_model // 4, out_dim))

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, emb_dim)

        Output:
            logp(x | noise, condition): (b, 1)
        """
        feat = super().forward(x, noise, condition).mean(1)
        return self.proj(feat)
