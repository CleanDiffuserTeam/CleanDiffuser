from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.utils import TensorDict, at_least_ndim, dict_apply


class FourierCondition(MLPCondition):
    """Fourier condition.

    Fourier condition supposes the input condition is of shape (b, 1).
    It produces fourier features with `in_dim` dimensions and then uses
    a MLP to output the features with `out_dim` dimensions.

    Args:
        in_dim (int):
            The input dimension of fourier features.
        out_dim (int):
            The output dimension of the condition.
        hidden_dims (int or List[int]):
            The hidden dimensions of the MLP
        act (nn.Module):
            The activation function. Default: nn.Mish()
        scale (float):
            The scale of the frequencies. Default: 16
        dropout (float):
            The dropout rate. Default: 0.25

    Examples:
        >>> nn_condition = FourierCondition(in_dim=5, out_dim=10, hidden_dims=[128, 128])
        >>> x = torch.randn((2, 1))
        >>> nn_condition(x).shape
        torch.Size([2, 10])
        >>> x = {
            "state1": torch.randn((2, 1)),
            "state2": torch.randn((2, 3, 1)),
        }
        >>> nn_condition(x)["state1"].shape
        torch.Size([2, 10])
        >>> nn_condition(x)["state2"].shape
        torch.Size([2, 3, 10])
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Union[int, List[int]],
        act=nn.Mish(),
        scale: float = 16,
        dropout: float = 0.25,
    ):
        super().__init__(in_dim, out_dim, hidden_dims, act, dropout)
        self.freqs = nn.Parameter(torch.randn(in_dim // 2) * scale, requires_grad=False)

    @staticmethod
    def mult_freqs(x: torch.Tensor, freqs: torch.Tensor):
        extended_freqs = 2 * np.pi * at_least_ndim(freqs, x.dim(), 1)
        emb = x * extended_freqs
        return torch.cat([emb.cos(), emb.sin()], -1)

    def forward(
        self, condition: Union[TensorDict, torch.Tensor], mask: Optional[Union[TensorDict, torch.Tensor]] = None
    ):
        condition = dict_apply(condition, self.mult_freqs, freqs=self.freqs)
        return super().forward(condition, mask)


class PositionalCondition(MLPCondition):
    """Positional condition.

    Positional condition supposes the input condition is of shape (b, 1).
    It produces positional features with `out_dim` dimensions and then uses
    a MLP to output the features with `out_dim` dimensions.

    Args:
        in_dim (int):
            The input dimension of positional features.
        out_dim (int):
            The output dimension of the condition.
        hidden_dims (int or List[int]):
            The hidden dimensions of the MLP
        act (nn.Module):
            The activation function. Default: nn.Mish()
        max_positions (int):
            The maximum number of positions. Default: 10000
        endpoint (bool):
            Whether to use endpoint. Default: False
        dropout (float):
            The dropout rate. Default: 0.25

    Examples:
        >>> nn_condition = PositionalCondition(in_dim=5, out_dim=10, hidden_dims=[128, 128])
        >>> x = torch.randint(10000, (2, 1))
        >>> nn_condition(x).shape
        torch.Size([2, 10])
        >>> x = {
            "state1": torch.randint(10000, (2, 1)),
            "state2": torch.randint(10000, (2, 3, 1)),
        }
        >>> nn_condition(x)["state1"].shape
        torch.Size([2, 10])
        >>> nn_condition(x)["state2"].shape
        torch.Size([2, 3, 10])
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Union[int, List[int]],
        act=nn.Mish(),
        max_positions: int = 10000,
        endpoint: bool = False,
        dropout: float = 0.25,
    ):
        super().__init__(in_dim, out_dim, hidden_dims, act, dropout)
        freqs = torch.arange(start=0, end=in_dim // 2, dtype=torch.float32)
        freqs = freqs / (in_dim // 2 - (1 if endpoint else 0))
        freqs = (1 / max_positions) ** freqs
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    @staticmethod
    def mult_freqs(x: torch.Tensor, freqs: torch.Tensor):
        extended_freqs = at_least_ndim(freqs, x.dim(), 1)
        emb = x * extended_freqs
        return torch.cat([emb.cos(), emb.sin()], -1)

    def forward(
        self, condition: Union[TensorDict, torch.Tensor], mask: Optional[Union[TensorDict, torch.Tensor]] = None
    ):
        condition = dict_apply(condition, self.mult_freqs, freqs=self.freqs)
        return super().forward(condition, mask)


if __name__ == "__main__":
    x1 = torch.randn((2, 1))
    x2 = {"s1": torch.randn((2, 1)), "s2": {"s3": torch.randn((2, 3, 1))}}

    m1 = FourierCondition(in_dim=6, out_dim=10, hidden_dims=[128, 128])
    m2 = PositionalCondition(in_dim=6, out_dim=10, hidden_dims=[128, 128])

    print(m1(x1).shape)
    print(m1(x2)["s1"].shape)
    print(m1(x2)["s2"]["s3"].shape)

    print(m2(x1).shape)
    print(m2(x2)["s1"].shape)
    print(m2(x2)["s2"]["s3"].shape)
