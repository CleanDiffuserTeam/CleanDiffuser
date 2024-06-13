from typing import List

import torch
import torch.nn as nn

from cleandiffuser.utils import Mlp
from .base_nn_classifier import BaseNNClassifier


class MLPNNClassifier(BaseNNClassifier):
    def __init__(
            self, x_dim: int, out_dim: int, emb_dim: int,
            hidden_dims: List[int], activation: nn.Module = nn.ReLU(), out_activation: nn.Module = nn.Identity(),
            timestep_emb_type: str = "positional"):
        super().__init__(emb_dim, timestep_emb_type)
        self.mlp = Mlp(
            in_dim=x_dim + emb_dim, hidden_dims=hidden_dims, out_dim=out_dim,
            activation=activation, out_activation=out_activation)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        out = self.mlp(torch.cat([x, self.map_noise(t)], dim=-1))
        return out
