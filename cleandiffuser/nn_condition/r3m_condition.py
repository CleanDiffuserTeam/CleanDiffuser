from typing import Optional

import torch
import torchvision.transforms as T

from cleandiffuser.nn_condition import BaseNNCondition


class R3MImageCondition(BaseNNCondition):
    """Pre-trained R3M model for image condition from https://github.com/facebookresearch/r3m.

    R3M first resizes the image to 224x224 and then encodes it with pre-trained resnet models.
    It expects the image to be in the range [0, 255].

    **Note:** You have to install `r3m` before using this module. Please refer to https://github.com/facebookresearch/r3m for installation instructions.

    Args:
        modelid (str):
            Pre-trained model name. One of "resnet18", "resnet34", "resnet50".
            Both "resnet18" and "resnet34" output representations of size 512, while "resnet50" outputs 2048.
        freeze (bool):
            Whether to freeze the model. Default is True.

    Example:
        >>> nn_condition = R3MImageCondition(modelid="resnet50")
        >>> image = torch.randint(0, 256, (2, 3, 400, 400), dtype=torch.uint8)
        >>> nn_condition(image).shape
        torch.Size([2, 2048])
        >>> image = torch.randint(0, 256, (2, 5, 6, 3, 200, 200), dtype=torch.uint8)
        >>> nn_condition(image).shape
        torch.Size([2, 5, 6, 2048])
    """

    def __init__(
        self,
        modelid: str = "resnet50",  # resnet18, resnet34, resnet50
        freeze: bool = True,
    ):
        super().__init__()
        from r3m import load_r3m

        rep = load_r3m(modelid)

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
