from typing import List, Optional

import torch
import torch.nn as nn

from .base_nn_diffusion import BaseNNDiffusion


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, emb_dim: int):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU())
        self.linear2 = nn.Sequential(nn.Linear(out_dim, out_dim), nn.SiLU())
        self.linearc = nn.Linear(emb_dim, out_dim)

        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        return self.linear2(self.linear1(x) + self.linearc(c)) + self.skip(x)


class SfBCUNet(BaseNNDiffusion):
    """Mlp-structured U-Net diffusion model backbone.

    A U-Net built with MLPs. It is designed in SfBC to generate actions.

    Args:
        x_dim (int):
            The dimension of the input. It is referred to as the dimension of `action` in SfBC.
        emb_dim (int):
            The dimension of the timestep embedding and condition embedding. 
            The condition is referred to as `observation` in SfBC. It should be in the shape of (b, emb_dim).
        hidden_dims (List[int]):
            The hidden dimensions of the mlp.
        timestep_emb_type (str):
            The type of the timestep embedding. Default: "positional"
        timestep_emb_params (dict):
            The parameters of the timestep embedding. Default: None

    Examples:
        >>> model = SfBCUNet(x_dim=10, emb_dim=16, hidden_dims=[256, 128, 64])
        >>> x = torch.randn((2, 10))
        >>> t = torch.randint(1000, (2,))
        >>> condition = torch.randn((2, 16))  # Should be of shape (b, emb_dim)
        >>> model(x, t, condition).shape
        torch.Size([2, 10])
        >>> model(x, t, None).shape
        torch.Size([2, 10])
    """

    def __init__(
        self,
        x_dim: int,
        emb_dim: int = 64,
        hidden_dims: List[int] = (512, 256, 128),
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        n_layers = len(hidden_dims)

        self.t_layer = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        in_dim = x_dim
        for i in range(n_layers):
            self.down_blocks.append(ResidualBlock(in_dim, hidden_dims[i], emb_dim))
            in_dim = hidden_dims[i]

        self.mid_block = ResidualBlock(in_dim, in_dim, emb_dim)

        for i in range(n_layers - 1):
            self.up_blocks.append(ResidualBlock(in_dim + hidden_dims[-1 - i], hidden_dims[-2 - i], emb_dim))
            in_dim = hidden_dims[-2 - i]

        self.out_layer = nn.Linear(in_dim, x_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None):
        c = self.t_layer(self.map_noise(t))
        if condition is not None:
            c += condition

        buffer = []
        for block in self.down_blocks:
            x = block(x, c)
            buffer.append(x)

        x = self.mid_block(x, c)

        for block in self.up_blocks:
            x = torch.cat([x, buffer.pop()], dim=-1)
            x = block(x, c)

        return self.out_layer(x)
