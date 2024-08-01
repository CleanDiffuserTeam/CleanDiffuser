from typing import Optional

import numpy as np

from .utils import at_least_ndim


class EmptyNormalizer:
    """ Empty Normalizer

    Does nothing to the input data.
    """

    def normalize(self, x: np.ndarray):
        return x

    def unnormalize(self, x: np.ndarray):
        return x


class GaussianNormalizer(EmptyNormalizer):
    """ Gaussian Normalizer

    Normalizes data to have zero mean and unit variance.
    For those dimensions with zero variance, the normalized value will be zero.

    Args:
        X: np.ndarray,
            dataset with shape (..., *x_shape)
        start_dim: int,
            the dimension to start normalization from, Default: -1

    Examples:
        >>> x_dataset = np.random.randn(100000, 3, 10)

        >>> normalizer = GaussianNormalizer(x_dataset, 1)
        >>> x = np.random.randn(1, 3, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)

        >>> normalizer = GaussianNormalizer(x_dataset, 2)
        >>> x = np.random.randn(1, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)
    """

    def __init__(self, X: np.ndarray, start_dim: int = -1):
        total_dims = X.ndim
        if start_dim < 0:
            start_dim = total_dims + start_dim

        axes = tuple(range(start_dim))

        self.mean = np.mean(X, axis=axes)
        self.std = np.std(X, axis=axes)
        self.std[self.std == 0] = 1.

    def normalize(self, x: np.ndarray):
        ndim = x.ndim
        return (x - at_least_ndim(self.mean, ndim, 1)) / at_least_ndim(self.std, ndim, 1)

    def unnormalize(self, x: np.ndarray):
        ndim = x.ndim
        return x * at_least_ndim(self.std, ndim, 1) + at_least_ndim(self.mean, ndim, 1)


class MinMaxNormalizer(EmptyNormalizer):
    """ MinMax Normalizer

    Normalizes data from range [min, max] to [-1, 1].
    For those dimensions with zero range, the normalized value will be zero.

    Args:
        X: np.ndarray,
            dataset with shape (..., *x_shape)
        start_dim: int,
            the dimension to start normalization from, Default: -1
        X_max: Optional[np.ndarray],
            Maximum value for each dimension. If None, it will be calculated from X. Default: None
        X_min: Optional[np.ndarray],
            Minimum value for each dimension. If None, it will be calculated from X. Default: None

    Examples:
        >>> x_dataset = np.random.randn(100000, 3, 10)

        >>> x_min = np.random.randn(3, 10)
        >>> normalizer = MinMaxNormalizer(x_dataset, 1, X_min=x_min)
        >>> x = np.random.randn(1, 3, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)

        >>> x_max = np.random.randn(10)
        >>> normalizer = MinMaxNormalizer(x_dataset, 2, X_max=x_max)
        >>> x = np.random.randn(1, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)
    """

    def __init__(
            self, X: np.ndarray, start_dim: int = -1,
            X_max: Optional[np.ndarray] = None, X_min: Optional[np.ndarray] = None):
        total_dims = X.ndim
        if start_dim < 0:
            start_dim = total_dims + start_dim

        axes = tuple(range(start_dim))

        self.max = np.max(X, axis=axes) if X_max is None else X_max
        self.min = np.min(X, axis=axes) if X_min is None else X_min
        self.mask = np.ones_like(self.max)
        self.range = self.max - self.min
        self.mask[self.max == self.min] = 0.
        self.range[self.range == 0] = 1.

    def normalize(self, x: np.ndarray):
        ndim = x.ndim
        x = (x - at_least_ndim(self.min, ndim, 1)) / at_least_ndim(self.range, ndim, 1)
        x = x * 2 - 1
        x = x * at_least_ndim(self.mask, ndim, 1)
        return x

    def unnormalize(self, x: np.ndarray):
        ndim = x.ndim
        x = (x + 1) / 2
        x = x * at_least_ndim(self.mask, ndim, 1)
        x = x * at_least_ndim(self.range, ndim, 1) + at_least_ndim(self.min, ndim, 1)
        return x
