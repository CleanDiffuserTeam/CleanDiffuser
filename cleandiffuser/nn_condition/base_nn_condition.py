import torch
import torch.nn as nn

from cleandiffuser.utils import at_least_ndim


def get_mask(mask: torch.Tensor, mask_shape: tuple, dropout: float, train: bool, device: torch.device):
    if train:
        mask = (torch.rand(mask_shape, device=device) > dropout).float()
    else:
        mask = 1. if mask is None else mask
    return mask


class BaseNNCondition(nn.Module):
    """
    In decision-making tasks, generating condition selections can be very diverse,
    including cumulative rewards, languages, images, demonstrations, and so on.
    It can even be a combination of these conditions. Therefore, we aim for
    nn_condition to handle diverse condition selections flexibly and
    ultimately output a tensor of shape (b, *cond_out_shape).

    Input:
        - condition: Any
        - mask : Optional, (b, ) or None, None means no mask

    Output:
        - condition: (b, *cond_out_shape)
    """

    def __init__(self, ):
        super().__init__()

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        raise NotImplementedError


class IdentityCondition(BaseNNCondition):
    """
    Identity condition does not change the input condition.

    Input:
        - condition: (b, *cond_in_shape)
        - mask :     (b, ) or None, None means no mask

    Output:
        - condition: (b, *cond_in_shape)
    """

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        mask = at_least_ndim(get_mask(
            mask, (condition.shape[0],), self.dropout, self.training, condition.device), condition.dim())
        return condition * mask
