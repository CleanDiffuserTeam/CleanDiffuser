from typing import Optional

import torch
import torchvision.transforms as T
from vip import load_vip

from cleandiffuser.nn_condition import BaseNNCondition


class VIPImageCondition(BaseNNCondition):
    """Pre-trained VIP model for image condition from https://github.com/facebookresearch/vip.

    VIP first resizes the image to 224x224 and then encodes it with pre-trained resnet50 models.
    It expects the image to be in the range [0, 255]. VIP outputs representations of size 1024.

    **Note:** You have to install `vip` before using this module. Please refer to https://github.com/facebookresearch/vip for installation instructions.

    Args:
        freeze (bool):
            Whether to freeze the model. Default is True.

    Example:
        >>> nn_condition = VIPImageCondition()
        >>> image = torch.randint(0, 256, (2, 3, 400, 400), dtype=torch.uint8)
        >>> nn_condition(image).shape
        torch.Size([2, 1024])
        >>> image = torch.randint(0, 256, (2, 5, 6, 3, 200, 200), dtype=torch.uint8)
        >>> nn_condition(image).shape
        torch.Size([2, 5, 6, 1024])
    """

    def __init__(
        self,
        freeze: bool = True,
    ):
        super().__init__()
        rep = load_vip("resnet50")

        self.model = rep.module

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.resize_transform = T.Resize(224)

    def forward(self, condition: torch.Tensor, mask: Optional[torch.Tensor] = None):
        leading_dims = condition.shape[:-3]
        image_dims = condition.shape[-3:]
        condition = condition.reshape(-1, *image_dims)
        condition = self.resize_transform(condition)
        return self.model(condition).reshape(*leading_dims, -1)
