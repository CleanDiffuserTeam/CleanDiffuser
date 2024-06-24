from typing import Optional, List

import torch
import torch.nn as nn

from cleandiffuser.utils import at_least_ndim, GroupNorm1d
from cleandiffuser.nn_diffusion import BaseNNDiffusion



class FCBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            GroupNorm1d(out_dim, 1, out_dim), nn.GELU())

    def forward(self, x):
        return self.model(x)


class EmbeddingBlock(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim))

    def forward(self, x):
        return self.model(x)


class MLPSieve(BaseNNDiffusion):
    """
    MLPSieve nn backbone proposed in the paper "Imitating Human Behavior with Diffusion Models"
    """

    def __init__(
            self, a_dim: int, history_len: int,
            hidden_dim: int = 512, emb_dim: int = 128,
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.act_emb = EmbeddingBlock(a_dim, emb_dim)
        self.fc1 = FCBlock(emb_dim * (3 + history_len), hidden_dim)
        self.fc2 = FCBlock(hidden_dim + a_dim + 1, hidden_dim)
        self.fc3 = FCBlock(hidden_dim + a_dim + 1, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim + a_dim + 1, a_dim)

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[List[torch.Tensor]] = None):
        """
        Input:
            x:          (b, horizon, in_dim)
            noise:      (b, )
            condition:  (b, history_len * emb_dim) or None / No condition indicates zeros((b, history_len * emb_dim))

        Output:
            y:          (b, horizon, in_dim)
        """
        x_e, emb = self.act_emb(x), self.map_noise(noise)
        noise = at_least_ndim(noise, 2)
        nn1 = self.fc1(torch.cat([x_e, emb, condition], -1))
        nn2 = self.fc2(torch.cat([nn1 / 1.414, x, noise], -1)) + nn1 / 1.414
        nn3 = self.fc3(torch.cat([nn2 / 1.414, x, noise], -1)) + nn2 / 1.414
        return self.fc4(torch.cat([nn3, x, noise], -1))
