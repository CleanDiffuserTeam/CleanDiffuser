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
        self.model = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            GroupNorm1d(out_feats, 8, 4),
            nn.GELU())

    def forward(self, x):
        return self.model(x)


class PearceMlp(BaseNNDiffusion):
    def __init__(
            self, act_dim: int, To: int = 1, timestep_emb_type: str = "positional",
            emb_dim: int = 128, hidden_dim: int = 512, timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.act_emb = nn.Sequential(
            nn.Linear(act_dim, emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))

        self.fcs = nn.ModuleList([
            FCBlock(emb_dim * (2 + To), hidden_dim),
            FCBlock(hidden_dim + act_dim + 1, hidden_dim),
            FCBlock(hidden_dim + act_dim + 1, hidden_dim),
            nn.Linear(hidden_dim + act_dim + 1, act_dim)])

        self.To = To
        self.emb_dim = emb_dim

    def forward(self,
                x: torch.Tensor, noise: torch.Tensor,
                condition: Optional[torch.Tensor] = None):
        """
        Input:
            x:          (b, act_dim)
            noise:      (b, )
            condition:  (b, To, emb_dim)

        Output:
            y:          (b, act_dim)
        """
        x_e, t_e = self.act_emb(x), self.map_noise(noise)
        t = noise.unsqueeze(-1)

        if condition is not None:
            nn1 = self.fcs[0](torch.cat([x_e, t_e, torch.flatten(condition, 1)], -1))
        else:
            condition = torch.zeros(x.shape[0], self.To, self.emb_dim).to(x.device)
            nn1 = self.fcs[0](torch.cat([x_e, t_e, torch.flatten(condition, 1)], -1))
        nn2 = self.fcs[1](torch.cat([nn1 / 1.414, x, t], -1)) + nn1 / 1.414
        nn3 = self.fcs[2](torch.cat([nn2 / 1.414, x, t], -1)) + nn2 / 1.414
        out = self.fcs[3](torch.cat([nn3, x, t], -1))

        return out
