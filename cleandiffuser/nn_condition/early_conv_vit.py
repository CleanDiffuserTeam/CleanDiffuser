from typing import Tuple, Optional, Dict

import einops
import torch
import torch.nn as nn

from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.utils import SinusoidalEmbedding, Transformer


class SmallStem(nn.Module):
    def __init__(
            self, patch_size: int = 16,
            in_channels: int = 3,
            channels_per_group: int = 16,
            kernel_sizes: tuple = (3, 3, 3, 3),
            strides: tuple = (2, 2, 2, 2),
            features: tuple = (32, 64, 128, 256),
            padding: tuple = (1, 1, 1, 1),
            num_features: int = 256,
    ):
        super().__init__()

        self.patch_size = patch_size

        cnn = []
        for n, (k, s, f, p) in enumerate(zip(kernel_sizes, strides, features, padding)):
            cnn.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels if n == 0 else features[n - 1], f, k, s, p),
                    nn.GroupNorm(f // channels_per_group, f),
                    nn.ReLU()))
        self.cnn = nn.Sequential(*cnn)

        self.patchify = nn.Conv2d(
            features[-1], num_features,
            kernel_size=patch_size // 16,
            stride=patch_size // 16, padding=0)

    def forward(self, x):
        x = self.cnn(x)
        x = self.patchify(x)
        return einops.rearrange(x, "b c h w -> b (h w) c")


class EarlyConvViTMultiViewImageCondition(BaseNNCondition):
    """ Early-CNN Vision Transformer (ViT) for multi-view image condition.

    A ViT model that uses a shallow CNN instead of a patchify layer to extract image tokens.
    This architecture is proposed in https://arxiv.org/pdf/2106.14881 and demonstrated to be
    effective for CV tasks. The vision encoder in Octo (https://arxiv.org/pdf/2405.12213) is
    mainly based on this architecture.
    Each view is processed by a separate CNN and the resulting tokens are concatenated along
    the token dimension, which are then processed by a transformer. The output of the
    learnable 'readout' token is returned as the final representation.

    Args:
        image_sz: Tuple[int],
            The size of the input image for each view. Assumes square images.
        in_channels: Tuple[int],
            The number of input channels for each view.
        lowdim_sz: Optional[int],
            The size of the low-dimensional condition. If None, no low-dimensional condition is used.
        To: int,
            The number of frames for each view.

        # Transformer arguments
        d_model: int:
            The dimension of the transformer token.
        nhead: int:
            The number of heads in the transformer.
        num_layers: int:
            The number of transformer layers.
        attn_dropout: float:
            The dropout rate for the attention layer.
        ffn_dropout: float:
            The dropout rate for the feedforward layer.

        # CNN arguments
        patch_size: Tuple[int]:
            The size of the patch for each view.
        channels_per_group: Tuple[int]:
            The number of channels per group in the CNN.
        kernel_sizes: Tuple[Tuple[int]]:
            The kernel sizes for each CNN layer.
        strides: Tuple[Tuple[int]]:
            The strides for each CNN layer.
        features: Tuple[Tuple[int]]:
            The number of features for each CNN layer.
        padding: Tuple[Tuple[int]]:
            The padding for each CNN layer.

    Examples:
        >>> d_model = 384
        >>> batch, view, To, C, H, W, D = 4, 2, 1, 3, 64, 64, 9
        >>> nn_condition = EarlyConvViTMultiViewImageCondition(d_model=d_model, ...)
        >>> condition = {
        ...     "image": torch.randn((batch, view, To, C, H, W)),
        ...     "lowdim": torch.randn((batch, To, D)),}
        >>> nn_condition(condition).shape
        torch.Size([batch, d_model])
    """
    def __init__(
            self,
            image_sz: Tuple[int] = (64, 64),
            in_channels: Tuple[int] = (3, 3),
            lowdim_sz: Optional[int] = None,
            To: int = 1,

            # Transformer parameters
            d_model: int = 384,
            nhead: int = 6,
            num_layers: int = 2,
            attn_dropout: float = 0.,
            ffn_dropout: float = 0.,

            # CNN parameters
            patch_size: Tuple[int] = (16, 16),
            channels_per_group: Tuple[int] = (16, 16),
            kernel_sizes: Tuple[Tuple[int]] = ((3, 3, 3, 3), (3, 3, 3, 3)),
            strides: Tuple[Tuple[int]] = ((2, 2, 2, 2), (2, 2, 2, 2)),
            features: Tuple[Tuple[int]] = ((32, 64, 128, 256), (32, 64, 128, 256)),
            padding: Tuple[Tuple[int]] = ((1, 1, 1, 1), (1, 1, 1, 1)),
    ):
        super().__init__()

        self.image_sz, self.in_channels = image_sz, in_channels
        self.n_views = len(image_sz)

        self.patchifies = nn.ModuleList([
            SmallStem(
                patch_size=patch_size[i],
                in_channels=in_channels[i],
                channels_per_group=channels_per_group[i],
                kernel_sizes=kernel_sizes[i],
                strides=strides[i],
                features=features[i],
                padding=padding[i],
                num_features=d_model) for i in range(self.n_views)])

        self.pos_emb = nn.ParameterList([
            nn.Parameter(
                SinusoidalEmbedding(d_model)(torch.arange(To * self.image_token_lens[i]))[None, :],
                requires_grad=False)
            for i in range(self.n_views)])

        self.view_emb = nn.ParameterList([
            nn.Parameter(
                torch.zeros((1, 1, d_model)), requires_grad=True)
            for _ in range(self.n_views)])

        self.lowdim_proj = nn.Linear(lowdim_sz, d_model) if lowdim_sz is not None else None
        self.lowdim_emb = nn.Parameter(
            torch.zeros((1, 1, d_model)), requires_grad=True) if lowdim_sz is not None else None

        self.readout_emb = nn.Parameter(
            torch.zeros((1, 1, d_model)), requires_grad=True)

        self.tfm = Transformer(
            d_model, nhead, num_layers, 4, attn_dropout, ffn_dropout)

        self.mask_cache = None

    @property
    def image_token_lens(self):
        examples = [
            torch.randn((1, self.in_channels[i], self.image_sz[i], self.image_sz[i]))
            for i in range(self.n_views)]
        return [self.patchifies[i](examples[i]).shape[1] for i in range(self.n_views)]

    def forward(self, condition: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None):

        b, v, t, c, h, w = condition["image"].shape

        tokens = []

        if self.lowdim_proj is not None:
            tokens.append(
                self.lowdim_proj(condition["lowdim"]) + self.lowdim_emb)

        for i in range(v):
            view_tokens = self.patchifies[i](
                einops.rearrange(condition["image"][:, i], "b t c h w -> (b t) c h w"))
            view_tokens = (einops.rearrange(view_tokens, "(b t) n d -> b (t n) d", b=b)
                           + self.view_emb[i] + self.pos_emb[i])
            tokens.append(view_tokens)

        tokens.append(self.readout_emb.repeat(b, 1, 1))

        tokens = torch.cat(tokens, dim=1)

        if self.mask_cache is None or tokens.shape[1] != self.mask_cache.shape[1]:
            self.mask_cache = torch.tril(
                torch.ones(tokens.shape[1], tokens.shape[1], device=condition["image"].device), diagonal=0)

        return self.tfm(tokens, self.mask_cache)[0][:, -1]
