from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion

__all__ = ["DQLMlp"]


class DQLMlp(BaseNNDiffusion):
    """Mlp diffusion model backbone for Diffusion Q-Learning (DQL).

    A simple mlp that used in DQL.

    Args:
        x_dim (int):
            The dimension of the input. It is referred to as the dimension of `action` in DQL.
        emb_dim (int):
            The dimension of the timestep embedding and condition embedding.
            The condition is referred to as `observation` in DQL. It should be in the shape of (b, emb_dim).
        timestep_emb_type (str):
            The type of the timestep embedding. Default: "positional"
        timestep_emb_params (Optional[dict]):
            The parameters of the timestep embedding. Default: None

    Examples:
        >>> model = DQLMlp(x_dim=10, emb_dim=16, hidden_dims=256)
        >>> x = torch.randn((2, 10))
        >>> t = torch.randint(1000, (2,))
        >>> condition = torch.randn((2, 16))
        >>> model(x, t, condition).shape
        torch.Size([2, 10])
        >>> model(x, t, None).shape
        torch.Size([2, 10])
    """

    def __init__(
        self,
        x_dim: int,
        emb_dim: int = 16,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.Mish(), nn.Linear(emb_dim * 2, emb_dim)
        )

        self.mid_layer = nn.Sequential(
            nn.Linear(x_dim + emb_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
        )

        self.final_layer = nn.Linear(256, x_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None):
        t = self.time_mlp(self.map_noise(t))
        if condition is not None:
            t += condition
        x = torch.cat([x, t], -1)
        x = self.mid_layer(x)

        return self.final_layer(x)
