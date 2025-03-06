from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion

__all__ = ["IDQLMlp"]

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x):
        return x + self.net(x)


class IDQLMlp(BaseNNDiffusion):
    """Residual mlp diffusion model backbone.

    A well-designed mlp used in Implicit Diffusion Q-Learning (IDQL). It has residual structures and uses LayerNorm.

    Args:
        x_dim (int):
            The dimension of the input. It is referred to as the dimension of `action` in IDQL.
        emb_dim (int):
            The dimension of the embedding and condition embedding.
            The condition is referred to as `observation` in IDQL. It should be in the shape of (b, emb_dim).
        hidden_dim (int):
            The dimension of the hidden layers. Default: 256
        n_blocks (int):
            The number of residual blocks. Default: 3
        dropout (float):
            The dropout rate. Default: 0.1
        timestep_emb_type (str):
            The type of the timestep embedding. Default: "positional"
        timestep_emb_params (Optional[dict]):
            The parameters of the timestep embedding.

    Examples:
        >>> model = IDQLMlp(x_dim=10, emb_dim=16, hidden_dim=256, n_blocks=3, dropout=0.1)
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
        emb_dim: int = 64,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)

        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2), nn.Mish(), nn.Linear(emb_dim * 2, emb_dim))

        self.affine_in = nn.Linear(x_dim + emb_dim, hidden_dim)

        self.ln_resnet = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])

        self.affine_out = nn.Sequential(nn.Mish(), nn.Linear(hidden_dim, x_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None):
        emb = self.time_mlp(self.map_noise(t))
        if condition is not None:
            emb += condition
        x = torch.cat([x, emb], -1)
        x = self.affine_in(x)
        x = self.ln_resnet(x)
        return self.affine_out(x)
    
