from typing import Literal, Optional, Tuple, Union

import torch
from timm.models import VisionTransformer

from .base_nn_condition import IdentityCondition


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
