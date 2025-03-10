from typing import List, Optional, Union

import torch
import torch.nn as nn

from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import Mlp

__all__ = ["MlpNNDiffusion"]

class MlpNNDiffusion(BaseNNDiffusion):
    """Mlp diffusion model backbone.

    A simple mlp that concatenates the input and the timestep embedding to predict the output.

    Args:
        x_dim (int):
            The dimension of the input.
        emb_dim (int):
            The dimension of the timestep embedding and condition embedding.
        hidden_dims (Union[List[int], int]):
            The hidden dimensions of the mlp.
        activation (nn.Module):
            The activation function. Default: nn.ReLU()
        timestep_emb_type (str):
            The type of the timestep embedding. Default: "positional"
        timestep_emb_params (Optional[dict]):
            The parameters of the timestep embedding. Default: None

    Examples:
        >>> model = MlpNNDiffusion(x_dim=10, emb_dim=16, hidden_dims=256)
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
        emb_dim: int = 16,
        hidden_dims: Union[List[int], int] = (256, 256),
        activation: nn.Module = nn.ReLU(),
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, (int, float)) else hidden_dims
        self.mlp = Mlp(x_dim + emb_dim, hidden_dims, x_dim, activation)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None):
        t = self.map_noise(t)
        if condition is not None:
            t += condition
        return self.mlp(torch.cat([x, t], -1))
