from typing import Optional, Union

import torch
import torch.nn as nn

from cleandiffuser.utils import TensorDict, dict_operation, get_mask

# def get_mask(mask: torch.Tensor, mask_shape: tuple, dropout: float, train: bool, device: torch.device):
#     if train:
#         mask = (torch.rand(mask_shape, device=device) > dropout).float()
#     else:
#         mask = 1.0 if mask is None else mask
#     return mask


class BaseNNCondition(nn.Module):
    """
    In decision-making tasks, conditions of generation can be very diverse,
    including cumulative rewards, language instructions, images, demonstrations, and so on.
    They can even be combinations of these conditions. Therefore, we aim for
    `nn_condition` to handle diverse condition selections flexibly and
    ultimately output a `torch.Tensor` or a `TensorDict`. `TensorDict` is a nested dictionary, where
    the keys are strings and the values are `torch.Tensor` or `TensorDict`.
    Although `TensorDict` provides much more flexibility, it must be combined with a specific `nn_diffusion` that
    can handle the `TensorDict`. So most of our implementations only use `torch.Tensor` as conditions.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self, condition: Union[TensorDict, torch.Tensor], mask: Optional[Union[TensorDict, torch.Tensor]] = None
    ):
        raise NotImplementedError


class IdentityCondition(BaseNNCondition):
    """
    Identity condition maps condition to itself.

    Examples:
        >>> nn_condition = IdentityCondition(dropout=0.25)
        >>> x = torch.randn((2, 3))
        >>> y = nn_condition(x)
        >>> y.shape
        torch.Size([2, 3])
        >>> x = {
            "image": torch.randn((2, 3, 64, 64)),
            "lowdim": torch.randn((2, 7)),
        }
        >>> y = nn_condition(x)
        >>> y["image"].shape
        torch.Size([2, 3, 64, 64])
        >>> y["lowdim"].shape
        torch.Size([2, 7])
    """

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout

    def forward(
        self, condition: Union[TensorDict, torch.Tensor], mask: Optional[Union[TensorDict, torch.Tensor]] = None
    ):
        if mask is None:
            prob = self.dropout if self.training else 0.0
            mask = get_mask(condition, prob, dims=0)

        return dict_operation(condition, mask, lambda x, y: x * y)
