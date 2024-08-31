from typing import List, Optional, Union

import torch
import torch.nn as nn

from cleandiffuser.utils import Mlp, TensorDict, dict_apply
from cleandiffuser.nn_condition import IdentityCondition


class LinearCondition(IdentityCondition):
    """A sample affine condition

    Use a linear layer to project the input condition to the desired dimension.

    Args:
        in_dim (int):
            The input dimension of the condition
        out_dim (int):
            The output dimension of the condition
        dropout (float):
            The label dropout rate, Default: 0.25

    Examples:
        >>> nn_condition = LinearCondition(in_dim=5, out_dim=10)
        >>> x = torch.randn((2, 5))
        >>> nn_condition(x).shape
        torch.Size([2, 10])
        >>> x = {
            "state1": torch.randn((2, 5)),
            "state2": torch.randn((2, 3, 5)),
        }
        >>> nn_condition(x)["state1"].shape
        torch.Size([2, 10])
        >>> nn_condition(x)["state2"].shape
        torch.Size([2, 3, 10])
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.25):
        super().__init__(dropout)
        self.affine = nn.Linear(in_dim, out_dim)

    def forward(
        self, condition: Union[TensorDict, torch.Tensor], mask: Optional[Union[TensorDict, torch.Tensor]] = None
    ):
        return super().forward(dict_apply(condition, self.affine), mask)


class MLPCondition(IdentityCondition):
    """A simple MLP condition

    Use a simple MLP to project the input condition to the desired dimension.

    Args:
        in_dim (int):
            The input dimension of the condition
        out_dim (int):
            The output dimension of the condition
        hidden_dims (int or List[int]):
            The hidden dimensions of the MLP
        act: nn.Module,
            The activation function of the MLP
        dropout: float,
            The label dropout rate

    Examples:
        >>> nn_condition = MLPCondition(in_dim=5, out_dim=10, hidden_dims=[128, 128])
        >>> x = torch.randn((2, 5))
        >>> nn_condition(x).shape
        torch.Size([2, 10])
        >>> x = {
            "state1": torch.randn((2, 5)),
            "state2": torch.randn((2, 3, 5)),
        }
        >>> nn_condition(x)["state1"].shape
        torch.Size([2, 10])
        >>> nn_condition(x)["state2"].shape
        torch.Size([2, 3, 10])
    """

    def __init__(
        self, in_dim: int, out_dim: int, hidden_dims: Union[int, List[int]], act=nn.LeakyReLU(), dropout: float = 0.25
    ):
        super().__init__(dropout)
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        self.mlp = Mlp(in_dim, hidden_dims, out_dim, act)

    def forward(
        self, condition: Union[TensorDict, torch.Tensor], mask: Optional[Union[TensorDict, torch.Tensor]] = None
    ):
        return super().forward(dict_apply(condition, self.mlp), mask)


class MLPSieveObsCondition(IdentityCondition):
    """MLPSieve observation condition.

    Use a MLP to project the observations to the embedding dimension.
    After projection, flatten the embedding.

    Args:
        o_dim: int,
            The dimension of the observation. Suppose the observation has shape (b, To, o_dim),
            where b is the batch size, To is the number of frames, and o_dim is the dimension of each frame.
        emb_dim: int,
            The dimension of the condition embedding. Default: 128
        hidden_dim: int,
            The hidden dimension of the MLP. Default: 512
        dropout: float,
            The label dropout rate. Default: 0.25

    Examples:
        >>> nn_condition = MLPSieveObsCondition(o_dim=3, emb_dim=128, hidden_dim=512)
        >>> obs = torch.randn(2, 10, 3)
        >>> nn_condition(obs).shape
        torch.Size([2, 1280])
    """

    def __init__(self, o_dim: int, emb_dim: int = 128, hidden_dim: int = 512, dropout: float = 0.25):
        super().__init__(dropout)
        self.mlp = Mlp(o_dim, [hidden_dim], emb_dim, nn.LeakyReLU())

    def forward(
        self, condition: Union[TensorDict, torch.Tensor], mask: Optional[Union[TensorDict, torch.Tensor]] = None
    ):
        return super().forward(dict_apply(condition, lambda x: self.mlp(x).flatten(1), mask))


if __name__ == "__main__":
    x1 = torch.randn((2, 3))
    x2 = {"s1": torch.randn((2, 3)), "s2": {"s3": torch.randn((2, 5, 3))}}

    m1 = LinearCondition(in_dim=3, out_dim=10)
    m2 = MLPCondition(in_dim=3, out_dim=10, hidden_dims=[128, 128])

    print(m1(x1).shape)
    print(m1(x2)["s1"].shape)
    print(m1(x2)["s2"]["s3"].shape)

    print(m2(x1).shape)
    print(m2(x2)["s1"].shape)
    print(m2(x2)["s2"]["s3"].shape)
