from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.utils import SUPPORTED_TIMESTEP_EMBEDDING


class BaseNNDiffusion(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        assert timestep_emb_type in SUPPORTED_TIMESTEP_EMBEDDING.keys()
        super().__init__()
        timestep_emb_params = timestep_emb_params or {}
        self.map_noise = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
            emb_dim, **timestep_emb_params
        )
        self._ignored_hparams = []

    def forward(
        self, x: torch.Tensor, noise: torch.Tensor, condition: Optional[torch.Tensor] = None
    ):
        raise NotImplementedError
