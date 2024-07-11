from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_condition import IdentityCondition, get_mask
from cleandiffuser.utils import at_least_ndim


class PearceObsCondition(IdentityCondition):
    """ Observation condition from DiffusionBC: https://arxiv.org/abs/2301.10677

    The model suggests using multi-frame observations as conditions.
    It encodes each frame of observation using the same MLP,
    then flattens them to create a condition embedding.

    --------------------------------------------
    Args:
    - obs_dim: int
        The dimension of the observation. Suppose the observation has shape (b, To, obs_dim),
        where b is the batch size, To is the number of frames, and obs_dim is the dimension of each frame.
    - emb_dim: int
        The dimension of the condition embedding. Default: 128
    - flatten: bool
        Whether to flatten the condition embedding. Default: False
    - dropout: float
        The label dropout rate. Default: 0.25

    --------------------------------------------
    Inputs:
    - obs: torch.Tensor
        The observation tensor with shape (b, To, obs_dim).
    - mask: Optional[torch.Tensor]:
        The label dropout mask that is used during training. If None, no mask is applied.

    Outputs:
    - embs: torch.Tensor
        The condition embedding with shape (b, To * emb_dim) if flatten is True, else (b, To, emb_dim)
    """
    def __init__(self, obs_dim: int, emb_dim: int = 128, flatten: bool = False, dropout: float = 0.25):
        super().__init__(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))
        self.flatten = flatten

    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        mask = at_least_ndim(get_mask(
            mask, (obs.shape[0],), self.dropout, self.training, obs.device),
            2 if self.flatten else 3)
        embs = self.mlp(obs)  # (b, To, emb_dim)
        embs = torch.flatten(embs, 1) if self.flatten else embs
        return embs * mask
