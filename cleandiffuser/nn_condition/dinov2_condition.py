from typing import Optional

import torch
import torchvision.transforms as T
from transformers import AutoModel

from cleandiffuser.nn_condition import BaseNNCondition


class DINOv2ImageCondition(BaseNNCondition):
    """Pre-trained DINO-v2 model for image condition.

    Four models are available:
    - "facebook/dinov2-small" (384 embedding dimensions)
    - "facebook/dinov2-base" (768 embedding dimensions)
    - "facebook/dinov2-large" (1024 embedding dimensions)
    - "facebook/dinov2-giant" (1536 embedding dimensions)

    The models follow a Transformer architecture, with a patch size of 14.
    For a 224x224 image, this results in 1 class token + 256 patch tokens.
    The models can accept larger images provided the image shapes are multiples of the patch size (14).
    If this condition is not verified, the model will crop to the closest smaller multiple of the patch size.
    The models accept images in the range [0, 1] or [0, 255] depending on `do_rescale`.

    Args:
        pretrained_model_name_or_path (str):
            Pre-trained model name or path.
        do_rescale (bool):
            Whether to rescale the image to [0, 1] from [0, 255]. Default is True.
            If False, the image is assumed to be in the range [0, 1].
        only_pooler_output (bool):
            Whether to only return the pooler output. Default is True.
            If False, the model will return a sequence of hidden states.
        freeze (bool):
            Whether to freeze the model. Default is True.

    Example:
        >>> nn_condition = DINOv2ImageCondition(pretrained_model_name_or_path="facebook/dinov2-base")
        >>> image = torch.randint(0, 256, (2, 4, 3, 224, 224), dtype=torch.uint8)
        >>> nn_condition(image).shape
        torch.Size([2, 4, 768])
        >>> nn_condition = DINOv2ImageCondition(pretrained_model_name_or_path="facebook/dinov2-small", only_pooler_output=False)
        >>> nn_condition(image).shape
        torch.Size([2, 4, 257, 384])
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "facebook/dinov2-base",
        do_rescale: bool = True,
        only_pooler_output: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self._do_rescale = do_rescale
        self._only_pooler_output = only_pooler_output
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def forward(self, condition: torch.Tensor, mask: Optional[torch.Tensor] = None):
        leading_dims = condition.shape[:-3]
        image_dims = condition.shape[-3:]
        condition = condition.reshape(-1, *image_dims)
        if self._do_rescale:
            condition = condition / 255.0
        condition = self.transform(condition)
        out = self.model(condition)
        if self._only_pooler_output:
            out = out["pooler_output"].reshape(*leading_dims, -1)
        else:
            out_shape = out["last_hidden_state"].shape[1:]
            out = out["last_hidden_state"].reshape(*leading_dims, *out_shape)
        return out
