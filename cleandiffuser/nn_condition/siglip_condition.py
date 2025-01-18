from typing import Optional

import torch
import torchvision.transforms as T

from cleandiffuser.nn_condition import IdentityCondition


class SiglipImageCondition(IdentityCondition):
    """Pretrained SigLIP model for image condition.

    SigLIP proposes to replace the loss function used in CLIP by a simple pairwise sigmoid loss.
    This results in better performance in terms of zero-shot classification accuracy on ImageNet.
    See https://huggingface.co/collections/google/siglip-659d5e62f0ae1a57ae0e83ba for all models.

    This class loads a pretrained huggingface SigLIP model and uses it to extract image features.
    It accepts both uint8 and float32 images in the range [0, 255] or [0, 1].
    And returns the patch features or the pooler output depending on the `use_pooler_output` parameter.

    Args:
        pretrained_model_name_or_path (str):
            Pretrained model name or path.
        use_pooler_output (bool):
            Whether to only return the pooler output. Default is False.
        freeze (bool):
            Whether to freeze the model. Default is True.
        dropout (float):
            Classifier-free guidance condition dropout rate. Default is 0.0.
        **kwargs:
            Additional keyword arguments for creating huggingface SigLIP model.

    Examples:
        >>> nn_condition = SiglipImageCondition()
        >>> x = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
        >>> nn_condition(x).shape
        torch.Size([2, 196, 768])

        >>> nn_condition = SiglipImageCondition(use_pooler_output=True)
        >>> x = torch.rand((2, 4, 3, 224, 224))
        >>> nn_condition(x).shape
        torch.Size([2, 4, 768])

    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "google/siglip-base-patch16-224",
        use_pooler_output: bool = False,
        freeze: bool = True,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(dropout)

        from transformers import SiglipVisionModel

        self.transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.model = SiglipVisionModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        self._use_pooler_output = use_pooler_output

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self._ignored_hparams.append("model")

    def forward(
        self,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        leading_dims = condition.shape[:-3]
        condition = condition.reshape(-1, *condition.shape[-3:])

        if condition.dtype == torch.uint8:
            dtype = next(self.model.parameters()).dtype
            condition = condition.to(dtype) / 255.0
        condition = self.transform(condition)

        out = self.model(condition)
        if self._use_pooler_output:
            out = out["pooler_output"].reshape(*leading_dims, -1)
        else:
            out = out["last_hidden_state"]
            out = out.reshape(*leading_dims, *out.shape[1:])

        return super().forward(out, mask)
