from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn

from cleandiffuser.nn_condition import BaseNNCondition, IdentityCondition
from cleandiffuser.utils import get_mask


def get_norm(channel: int, use_group_norm: bool = True, group_channels: int = 16):
    if use_group_norm:
        return nn.GroupNorm(channel // group_channels, channel, affine=True)
    else:
        return nn.BatchNorm2d(channel, affine=True)


def get_image_coordinates(h, w, normalise):
    x_range = torch.arange(w, dtype=torch.float32)
    y_range = torch.arange(h, dtype=torch.float32)
    if normalise:
        x_range = (x_range / (w - 1)) * 2 - 1
        y_range = (y_range / (h - 1)) * 2 - 1
    image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
    image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
    return image_x, image_y


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        downsample: bool = False,
        use_group_norm: bool = True,
        group_channels: int = 16,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        stride = 2 if downsample else 1

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            get_norm(out_channel, use_group_norm, group_channels),
            activation,
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            get_norm(out_channel, use_group_norm, group_channels),
        )

        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
                get_norm(out_channel, use_group_norm, group_channels),
            )
            if downsample
            else nn.Identity()
        )

    def forward(self, x):
        return self.cnn(x) + self.skip(x)


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax layer as described in [1].
    [1]: "Deep Spatial Autoencoders for Visuomotor Learning"
    Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel

    Adapted from https://github.com/gorosgobe/dsae-torch/blob/master/dsae.py
    """

    def __init__(self, temperature=None, normalise=True):
        super().__init__()
        self.temperature = (
            nn.Parameter(torch.ones(1)) if temperature is None else torch.tensor([temperature])
        )
        self.normalise = normalise

    def forward(self, x):
        n, c, h, w = x.size()
        spatial_softmax_per_map = nn.functional.softmax(
            x.view(n * c, h * w) / self.temperature, dim=1
        )
        spatial_softmax = spatial_softmax_per_map.view(n, c, h, w)

        # calculate image coordinate maps
        image_x, image_y = get_image_coordinates(h, w, normalise=self.normalise)
        # size (H, W, 2)
        image_coordinates = torch.cat((image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1)
        # send to device
        image_coordinates = image_coordinates.to(device=x.device)

        # multiply coordinates by the softmax and sum over height and width, like in [2]
        expanded_spatial_softmax = spatial_softmax.unsqueeze(-1)
        image_coordinates = image_coordinates.unsqueeze(0)
        out = torch.sum(expanded_spatial_softmax * image_coordinates, dim=[2, 3])
        # (N, C, 2)
        return out


class ResNet18(nn.Module):
    def __init__(
        self,
        image_sz: int,
        in_channel: int,
        emb_dim: int,
        act_fn=lambda: nn.ReLU(),
        use_group_norm: bool = True,
        group_channels: int = 16,
        use_spatial_softmax: bool = True,
    ):
        super().__init__()

        self.image_sz, self.in_channel = image_sz, in_channel

        self.cnn = nn.Sequential(
            # Initial convolution
            nn.Conv2d(in_channel, 64, 7, 2, 3, bias=False),
            get_norm(64, use_group_norm, group_channels),
            act_fn(),
            nn.MaxPool2d(3, 2, 1),
            # Residual blocks
            ResidualBlock(64, 64, False, use_group_norm, group_channels, act_fn()),
            ResidualBlock(64, 64, False, use_group_norm, group_channels, act_fn()),
            ResidualBlock(64, 128, True, use_group_norm, group_channels, act_fn()),
            ResidualBlock(128, 128, False, use_group_norm, group_channels, act_fn()),
            ResidualBlock(128, 256, True, use_group_norm, group_channels, act_fn()),
            ResidualBlock(256, 256, False, use_group_norm, group_channels, act_fn()),
            ResidualBlock(256, 512, True, use_group_norm, group_channels, act_fn()),
            ResidualBlock(512, 512, False, use_group_norm, group_channels, act_fn()),
            # Final pooling
            nn.AvgPool2d(7, 1, 0) if not use_spatial_softmax else SpatialSoftmax(None, True),
        )

        self.mlp = nn.Sequential(
            nn.Linear(np.prod(self.cnn_output_shape), emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    @property
    def cnn_output_shape(self):
        example = torch.zeros(
            (1, self.in_channel, self.image_sz, self.image_sz), device=self.device, dtype=self.dtype
        )
        return self.cnn(example).shape

    def forward(self, x):
        feat = self.cnn(x)
        return self.mlp(torch.flatten(feat, 1))


class ResNet18ImageCondition(BaseNNCondition):
    """ResNet18 for image condition.

    ResNet18Condition encodes the input image into a fixed-size embedding.
    The implementation is adapted from `DiffusionPolicy`.
    Compared to the original implementation, we replace `BatchNorm2d` with `GroupNorm`,
    and use a SpatialSoftmax instead of an average pooling layer.

    Args:
        image_sz (int):
            Size of the input image. The image is assumed to be square.
        in_channel (int):
            Number of input channels. 3 for RGB images.
        emb_dim (int):
            Dimension of the output embedding.

        act_fn (callable):
            Activation function to use in the network. Default is ReLU.
        use_group_norm (bool):
            Whether to use GroupNorm instead of BatchNorm. Default is True.
        group_channels (int):
            Number of channels per group in GroupNorm. Default is 16.
        use_spatial_softmax (bool):
            Whether to use SpatialSoftmax instead of average pooling. Default is True.

        dropout (float):
            Condition Dropout rate. Default is 0.0.

    Examples:
        >>> nn_condition = ResNet18ImageCondition(image_sz=64, in_channel=3, emb_dim=256)
        >>> condition = torch.randn(32, 3, 64, 64)
        >>> nn_condition(condition).shape
        torch.Size([32, 256])
        >>> condition = torch.randn(32, 4, 3, 64, 64)
        >>> nn_condition(condition).shape
        torch.Size([32, 4, 256])
    """

    def __init__(
        self,
        image_sz: int,
        in_channel: int,
        emb_dim: int,
        act_fn=lambda: nn.ReLU(),
        use_group_norm: bool = True,
        group_channels: int = 16,
        use_spatial_softmax: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.image_sz, self.in_channel = image_sz, in_channel

        self.resnet18 = ResNet18(
            image_sz,
            in_channel,
            emb_dim,
            act_fn,
            use_group_norm,
            group_channels,
            use_spatial_softmax,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def feature_map_shape(self):
        return self.resnet18.cnn_output_shape

    def forward(self, condition: torch.Tensor, mask: Optional[torch.Tensor] = None):
        other_dims = condition.shape[:-3]

        condition = condition.reshape(-1, self.in_channel, self.image_sz, self.image_sz)

        condition = self.resnet18(condition)

        if mask is None:
            prob = self.dropout if self.training else 0.0
            mask = get_mask(condition, prob, dims=0)

        return (condition * mask).reshape(*other_dims, -1)


class ResNet18MultiViewImageCondition(BaseNNCondition):
    """ResNet18 for multi-view image condition.

    ResNet18Condition encodes the input image into a fixed-size embedding.
    The implementation is adapted from `DiffusionPolicy`.
    Compared to the original implementation, we replace `BatchNorm2d` with `GroupNorm`,
    and use a SpatialSoftmax instead of an average pooling layer.
    The Multi-view version uses different ResNet18 networks for each view.

    Args:
        image_sz (int):
            Size of the input image. The image is assumed to be square.
        in_channel (int):
            Number of input channels. 3 for RGB images.
        emb_dim (int):
            Dimension of the output embedding.
        n_views (int):
            Number of views.

        act_fn (callable):
            Activation function to use in the network. Default is ReLU.
        use_group_norm (bool):
            Whether to use GroupNorm instead of BatchNorm. Default is True.
        group_channels (int):
            Number of channels per group in GroupNorm. Default is 16.
        use_spatial_softmax (bool):
            Whether to use SpatialSoftmax instead of average pooling. Default is True.

        flatten (bool):
            Whether to flatten the output. Default is False.

        dropout (float):
            Condition Dropout rate. Default is 0.0.

    Examples:
        >>> nn_condition = ResNet18MultiViewImageCondition(image_sz=64, in_channel=3, emb_dim=256, n_views=2)
        >>> condition = torch.randn(32, 2, 3, 64, 64)  # (bs, n_views, ..., c, h, w)
        >>> nn_condition(condition).shape
        torch.Size([32, 2, 256])
        >>> condition = torch.randn(32, 2, 4, 3, 64, 64)  # (bs, n_views, ..., c, h, w)
        >>> nn_condition(condition).shape
        torch.Size([32, 2, 4, 256])
    """

    def __init__(
        self,
        image_sz: int,
        in_channel: int,
        emb_dim: int,
        n_views: int,
        act_fn=lambda: nn.ReLU(),
        use_group_norm: bool = True,
        group_channels: int = 16,
        use_spatial_softmax: bool = True,
        flatten: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.n_views = n_views
        self.image_sz, self.in_channel = image_sz, in_channel
        self.flatten = flatten

        self.resnet18 = nn.ModuleList(
            [
                ResNet18(
                    image_sz,
                    in_channel,
                    emb_dim,
                    act_fn,
                    use_group_norm,
                    group_channels,
                    use_spatial_softmax,
                )
                for _ in range(n_views)
            ]
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def feature_map_shape(self):
        return self.resnet18.cnn_output_shape

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        emb = []

        for n in range(self.n_views):
            this_condition = condition[:, n]

            other_dims = this_condition.shape[:-3]

            this_condition = this_condition.reshape(
                -1, self.in_channel, self.image_sz, self.image_sz
            )

            this_condition = self.resnet18[n](this_condition)

            if mask is None:
                prob = self.dropout if self.training else 0.0
                mask = get_mask(this_condition, prob, dims=0)

            emb.append((this_condition * mask).reshape(*other_dims, -1))

        emb = torch.stack(emb, 1)
        if self.flatten:
            emb = emb.flatten(1)
        return emb


class ResNetImageCondition(IdentityCondition):
    """Resnet image condition.

    Use timm ResNet implementations for image encoding.
    These models can be either pretrained or not.
    Full configuration can be found in https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py.

    Args:
        model_name (str):
            Name of the ResNet model, e.g., 'resnet18', 'resnet50', etc.
        pretrained (bool):
            Whether to load the pretrained model. Default is False.
            If True, the model will be loaded with the pretrained weights.
            And do not forget to normalize the input image with the same normalization as the pretrained model.
        freeze (bool):
            Whether to freeze the model. It is useful when you want to use the model as a pretrained feature extractor.
        flatten (bool):
            Whether to flatten the output. Default is False.
        dropout (float):
            Classifier-free guidance condition dropout rate. Default is 0.0.
        **kwargs:
            Other keyword arguments to pass to the `timm.create_model` function.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = False,
        freeze: bool = False,
        flatten: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(dropout)
        self.flatten = flatten
        self.resnet = timm.create_model(model_name, pretrained=pretrained, **kwargs)
        if freeze:
            self.resnet = self.resnet.requires_grad_(False).eval()

        # freeze the last linear classification layer since we don't need it
        self.resnet.fc.requires_grad_(False)

    def forward(self, condition: torch.Tensor, mask: Optional[torch.Tensor] = None):
        leading_dims = condition.shape[:-3]
        condition = condition.reshape(-1, *condition.shape[-3:])
        img_feat = self.resnet.forward_features(condition)
        img_feat = self.resnet.global_pool(img_feat)
        img_feat = img_feat.reshape(*leading_dims, -1)
        if self.flatten:
            img_feat = img_feat.flatten(1)
        return super().forward(img_feat, mask)


if __name__ == "__main__":
    # Learn from scratch
    x1 = torch.rand((32, 2, 3, 224, 224))
    m1 = ResNetImageCondition(model_name="resnet18", pretrained=False)

    print(m1(x1).shape)

    # Pretrained (don't forget to use imagenet normalization)
    import torchvision.transforms as T
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    transform = T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    x2 = torch.rand((32, 2, 3, 224, 224))
    x2 = transform(x2)
    m2 = ResNetImageCondition(model_name="resnet18", pretrained=True, freeze=True)

    print(m2(x2).shape)
