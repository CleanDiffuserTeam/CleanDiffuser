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

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None):
        out = self.mlp(torch.cat([x, self.map_noise(t)], dim=-1))
        return out


class QGPONNClassifier(BaseNNClassifier):
    """ NN of energy guidance model f_phi in QGPO """
    def __init__(
            self, obs_dim: int, act_dim: int, emb_dim: int,
            hidden_dims: List[int],
            timestep_emb_type: str = "positional"):
        super().__init__(emb_dim, timestep_emb_type)
        self.obs_proj = nn.Linear(obs_dim, emb_dim)
        self.act_proj = nn.Linear(act_dim, emb_dim)
        self.mlp = Mlp(
            in_dim=3 * emb_dim, hidden_dims=hidden_dims, out_dim=1, activation=nn.SiLU())

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """

        Args:
            x: torch.Tensor,
                Noisy actions. Shape (batch_size, act_dim)
            t: torch.Tensor,
                Timestep. Shape (batch_size, )
            y: torch.Tensor,
                Observations. Shape (batch_size, obs_dim)

        Returns:
            f: torch.Tensor,
                Energy prediction. Shape (batch_size, 1)
        """
        y = self.obs_proj(y)
        x = self.act_proj(x)
        out = self.mlp(torch.cat([y, x, self.map_noise(t)], dim=-1))
        return torch.tanh(out / 10) * 10
