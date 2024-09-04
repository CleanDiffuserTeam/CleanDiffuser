from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import GroupNorm1d


class TimeSiren(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(TimeSiren, self).__init__()
        # just a fully connected NN with sin activations
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class FCBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        # one layer of non-linearities (just a useful building block to use below)
        self.model = nn.Sequential(nn.Linear(in_feats, out_feats), GroupNorm1d(out_feats, 8, 4), nn.GELU())

    def forward(self, x):
        return self.model(x)


class PearceMlp(BaseNNDiffusion):
    """Pearce MLP diffusion model backbone.

    A well-designed mlp used in diffusion behavior clone (DBC).

    Args:
        x_dim (int):
            The dimension of the input. It is referred to as the dimension of `action` in DBC.
        emb_dim (int):
            The dimension of the timestep embedding and condition embedding.
        condition_horizon (int):
            The horizon of the condition embedding.
            The condition should be of shape (b, condition_horizon, emb_dim) and is referred to as `observation` in DBC.
        hidden_dim (int):
            The dimension of the hidden layer.
        timestep_emb_type (str):
            The type of the timestep embedding.
        timestep_emb_params (Optional[dict]):
            The parameters of the timestep embedding. Default: None

    Examples:
        >>> model = PearsonMlp(x_dim=10, emb_dim=16, hidden_dim=256, condition_horizon=2)
        >>> x = torch.randn((2, 10))
        >>> t = torch.randint(1000, (2,))
        >>> condition = torch.randn((2, 2, 16))  # Should be of shape (b, condition_horizon, emb_dim)
        >>> model(x, t, condition).shape
        torch.Size([2, 10])
        >>> model(x, t, None).shape
        torch.Size([2, 10])
    """

    def __init__(
        self,
        x_dim: int,
        emb_dim: int = 128,
        condition_horizon: int = 1,
        hidden_dim: int = 512,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.act_emb = nn.Sequential(nn.Linear(x_dim, emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))

        self.fcs = nn.ModuleList(
            [
                FCBlock(emb_dim * (2 + condition_horizon), hidden_dim),
                FCBlock(hidden_dim + x_dim + 1, hidden_dim),
                FCBlock(hidden_dim + x_dim + 1, hidden_dim),
                nn.Linear(hidden_dim + x_dim + 1, x_dim),
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None):
        x_e, t_e = self.act_emb(x), self.map_noise(t)
        t = t.unsqueeze(-1)
        if condition is not None:
            nn1 = self.fcs[0](torch.cat([x_e, t_e, torch.flatten(condition, 1)], -1))
        else:
            nn1 = self.fcs[0](torch.cat([x_e, t_e], -1))
        nn2 = self.fcs[1](torch.cat([nn1 / 1.414, x, t], -1)) + nn1 / 1.414
        nn3 = self.fcs[2](torch.cat([nn2 / 1.414, x, t], -1)) + nn2 / 1.414
        out = self.fcs[3](torch.cat([nn3, x, t], -1))
        return out
