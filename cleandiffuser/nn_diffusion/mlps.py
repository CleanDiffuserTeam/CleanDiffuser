from typing import List, Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import Mlp


class MlpNNDiffusion(BaseNNDiffusion):
    def __init__(
        self, 
        x_dim: int,
        emb_dim: int = 16,
        hidden_dims: List[int] = (256, 256),
        activation: nn.Module = nn.ReLU(),
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.mlp = Mlp(
            x_dim + emb_dim, hidden_dims, x_dim, activation)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, x_dim)
            noise:      (b, )
            condition:  (b, emd_dim)

        Output:
            y:          (b, x_dim)
        """
        t = self.map_noise(noise)
        if condition is not None:
            t += condition
        else:
            t += torch.zeros_like(t)
        return self.mlp(torch.cat([x, t], -1))


        