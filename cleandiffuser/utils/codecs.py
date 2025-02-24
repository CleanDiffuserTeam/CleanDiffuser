from typing import Any, List, Sequence, Tuple

import numpy as np
from imagecodecs import jpeg_decode, jpeg_encode
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray, ensure_ndarray, ndarray_copy
from numcodecs.registry import register_codec

JPEG_MAX_DIMENSION = 655_000
JPEG_MIN_RANK = 2


def validate_axis_reduction(
    input_shape: Sequence[int], axis_reduction: Any = None
) -> Tuple[List[int], List[int], List[int]]:
    rank = len(input_shape)
    input_dimensions = tuple(range(rank))
    if rank < JPEG_MIN_RANK:
        raise ValueError(
            f"Invalid chunk size. Chunk size must have have length 2 or greater; got chunk size with length = {rank}"
        )

    if axis_reduction is None:
        # partition the "full" dimensions and the singleton dimensions
        fulls, singletons = [], []
        for idx, chunk in enumerate(input_shape):
            if chunk == 1:
                singletons.append(idx)
            else:
                fulls.append(idx)
        if len(fulls) < JPEG_MIN_RANK:
            raise ValueError(
                "Invalid chunk size. At least 2 chunk sizes must be greater than 1; got {input_shape}"
            )
        # default behavior is to put all but one of the full dimensions + all of the singletons in the
        # first axis, then the last full dimension in the second axis
        result = ((*fulls[:-1], *singletons), (fulls[-1],), ())

    else:
        result = tuple(tuple(axis) for axis in axis_reduction)

    if len(result) == 2:
        result = (*result, [])

    if len(result) != 3:
        raise ValueError(
            f"Axis reduction for jpeg compression must have length 2 or 3; got axis_reduction with length = {len(result)}"
        )

    reduction_unpacked = tuple(sorted([dim for axis in result for dim in axis]))
    if not reduction_unpacked == input_dimensions:
        raise ValueError(
            f"Invalid axis reduction. Axis reduction must contain {input_dimensions}. Got an axis reduction that contained {reduction_unpacked} instead."
        )

    return result


class jpeg(Codec):
    """Codec providing jpeg compression via imagecodecs.

    Parameters
    ----------
    quality : int
        Compression level.
    """

    codec_id = "jpeg"

    def __init__(self, input_shape, axis_reduction=None, quality=100):
        self.quality = quality
        self.input_shape = tuple(input_shape)
        self.axis_reduction = validate_axis_reduction(input_shape, axis_reduction)
        assert self.quality > 0 and self.quality <= 100 and isinstance(self.quality, int)
        super().__init__()

    def encode(self, buf):
        bufa = ensure_ndarray(buf)
        axis_reduction = self.axis_reduction

        if bufa.ndim < 2:
            raise ValueError(
                f"Invalid dimensionality of input array.\n Input must have dimensionality of at least 2; got {buf.ndim}"
            )
        if len(self.input_shape) != len(bufa.shape):
            raise ValueError(
                f"Invalid input size.\n Input must have dimensionality matching the input_shape parameter of this compressor, i.e. {self.input_shape}, which has a dimensionality of {len(self.input_shape)}.\n Got input with shape {bufa.shape} instead, which has a dimensionality of {len(bufa.shape)}."
            )
        if not all(chnk >= shpe for chnk, shpe in zip(self.input_shape, bufa.shape)):
            raise ValueError(
                f"Invalid input size. Input must be less than or equal to the input_shape parameter of this compressor, i.e. {self.input_shape}. Got input with shape {bufa.shape} instead"
            )
        new_shape = [
            np.prod([bufa.shape[dim] for dim in axis], dtype="int") for axis in axis_reduction
        ]
        tiled = bufa.reshape(new_shape)

        return jpeg_encode(tiled, level=self.quality)

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf)

        if out is not None:
            out = ensure_contiguous_ndarray(out)

        tiled = jpeg_decode(buf)
        return ndarray_copy(tiled, out)


register_codec(jpeg)
