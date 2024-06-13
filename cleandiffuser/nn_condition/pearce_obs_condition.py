import torch
import torch.nn as nn

from cleandiffuser.nn_condition import IdentityCondition, get_mask
from cleandiffuser.utils import at_least_ndim


class PearceObsCondition(IdentityCondition):
    def __init__(self, obs_dim: int, emb_dim: int = 128, flatten: bool = False, dropout: float = 0.25):
        super().__init__(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))
        self.flatten = flatten

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        mask = at_least_ndim(get_mask(
            mask, (obs.shape[0],), self.dropout, self.training, obs.device), 2)
        embs = self.mlp(obs)  # (b, To, emb_dim)
        if self.flatten:
            out = torch.flatten(embs, 1) * mask
        else:
            if isinstance(mask, float):
                out = embs * mask
            else:
                out = embs * mask.unsqueeze(1)
        return out

