from typing import Literal, Optional, Tuple, Union

import torch
import torchvision.transforms as T

from cleandiffuser.nn_condition.base_nn_condition import IdentityCondition


class ViTCondition(IdentityCondition):
    """Vision Transformer Condition.

    The backbone model is a timm ViT model from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py.

    Args:
        img_size (Union[int, Tuple[int, int]], optional):
            Size of the input image. Defaults to 224.
        patch_size (Union[int, Tuple[int, int]], optional):
            Size of the patch. Defaults to 16.
        in_chans (int, optional):
            Number of input channels. Defaults to 3.
        global_pool (Literal["", "avg", "avgmax", "max", "token", "map"], optional):
            Global pooling type.
            If "", no global pooling is applied, and the output shape is (batch, num_patches, embed_dim).
            If "token", it returns the [CLS] feature in shape (batch, embed_dim).
            Defaults to "token".
        embed_dim (int):
            Dimension of the embedding. Defaults to 768.
        depth (int):
            Number of transformer layers. Defaults to 12.
        num_heads (int):
            Number of attention heads. Defaults to 12.
        class_token (bool):
            Whether to use class token. Defaults to True.
        reg_tokens (int):
            Number of regular tokens. Defaults to 0.
        dropout (float):
            Classifier-free guidance condition dropout. Defaults to 0.0.
        **kwargs:
            Additional arguments for the timm ViT model.

    Examples:
        >>> nn_condition = ViTCondition(global_pool="token")
        >>> nn_condition(torch.randn(2, 3, 224, 224)).shape
        torch.Size([2, 768])
        >>> nn_condition(torch.randn(2, 5, 3, 224, 224)).shape
        torch.Size([2, 5, 768])

        >>> nn_condition = ViTCondition(global_pool="")
        >>> nn_condition(torch.randn(2, 3, 224, 224)).shape
        torch.Size([2, 196, 768])
        >>> nn_condition(torch.randn(2, 5, 3, 224, 224)).shape
        torch.Size([2, 5, 196, 768])
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        class_token: bool = True,
        reg_tokens: int = 0,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.global_pool = global_pool

        from timm.models import VisionTransformer

        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            reg_tokens=reg_tokens,
            class_token=class_token,
            **kwargs,
        )

        self.prefix_len = int(class_token) + reg_tokens

    def forward(self, condition: torch.Tensor, mask: Optional[torch.Tensor] = None):
        leading_dim = condition.shape[:-3]
        condition = condition.view(-1, *condition.shape[-3:])
        feat = self.vit(condition)
        if self.global_pool == "":
            feat = feat[:, self.prefix_len :]
        return super().forward(feat, mask).view(*leading_dim, *feat.shape[1:])


class PretrainedViTCondition(IdentityCondition):
    """Pretrained ViT model for image condition.

    Pretrained ViT model on ImageNet-21k from https://huggingface.co/google/vit-base-patch16-224-in21k.

    This class loads a pretrained huggingface ViT model and uses it to extract image features.
    It accepts both uint8 and float32 images in the range [0, 255] or [0, 1].
    And returns the patch features or the pooler output depending on the `use_pooler_output` parameter.

    Args:
        pretrained_model_name_or_path (str):
            Pretrained model name or path.
        use_pooler_output (bool):
            Whether to only return the pooler output. Default is False.
        require_cls_feature (bool):
            Whether to require the CLS feature. Default is False.
        freeze (bool):
            Whether to freeze the model. Default is True.
        dropout (float):
            Classifier-free guidance condition dropout rate. Default is 0.0.
        **kwargs:
            Additional keyword arguments for creating huggingface ViT model.

    Examples:
        >>> nn_condition = PretrainedViTCondition()
        >>> x = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
        >>> nn_condition(x).shape
        torch.Size([2, 196, 768])

        >>> nn_condition = PretrainedViTCondition(use_pooler_output=True)
        >>> x = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
        >>> nn_condition(x).shape
        torch.Size([2, 768])

        >>> nn_condition = PretrainedViTCondition(require_cls_feature=True)
        >>> x = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
        >>> nn_condition(x).shape
        torch.Size([2, 197, 768])
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "google/vit-base-patch16-224-in21k",
        use_pooler_output: bool = False,
        require_cls_feature: bool = False,
        freeze: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(dropout)

        from transformers import ViTModel

        self.transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.model = ViTModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        self._use_pooler_output = use_pooler_output
        self._require_cls_feature = require_cls_feature

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
            if not self._require_cls_feature:
                out = out[:, 1:]
            out = out.reshape(*leading_dims, *out.shape[1:])

        return super().forward(out, mask)
