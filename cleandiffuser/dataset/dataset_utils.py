from typing import Dict, Callable, List, Union
import numpy as np
import torch
import scipy.interpolate as interpolate
import numba
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
import cleandiffuser.dataset.rotation_conversions as rc
import functools


# -----------------------------------------------------------------------------#
# ------------------------------ SequenceSampler ------------------------------#
# -----------------------------------------------------------------------------#

# Original implemetation: https://github.com/real-stanford/diffusion_policy
# Observation Horizon: To|n_obs_steps
# Action Horizon: Ta|n_action_steps
# Prediction Horizon: T|horizon
# To = 3
# Ta = 4
# T = 6
# |o|o|o|
# | | |a|a|a|a|
# pad_before = 2
# pad_after = 3

@numba.jit(nopython=True)
def create_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0, pad_after: int = 0,
        debug: bool = True) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0  # episode start index
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]  # episode end index
        episode_length = end_idx - start_idx  # episode length

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert (start_offset >= 0)
                assert (end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


class SequenceSampler:
    def __init__(
            self,
            replay_buffer: ReplayBuffer,
            sequence_length: int,
            pad_before: int = 0,
            pad_after: int = 0,
            keys=None,
            key_first_k=dict(),
            zero_padding: bool = False,
    ):
        """
            key_first_k: dict str: int
                Only take first k data from these keys (to improve perf)
        """
        super().__init__()
        assert (sequence_length >= 1)

        # all keys
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]

        # create indices
        # indices (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        # buffer_start_idx and buffer_end_idx define the actual start and end positions of the sample sequence within the original dataset.
        # sample_start_idx and sample_end_idx define the relative start and end positions within the sample sequence, 
        # which is particularly useful when dealing with padding as it can affect the actual length of the sequence.
        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.zero_padding = zero_padding
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:],
                                 fill_value=np.nan, dtype=input_arr.dtype)
                sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx + k_data]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if not self.zero_padding:
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < self.sequence_length:
                        data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result


# -----------------------------------------------------------------------------#
# ---------------------------- Rotation Transformer ---------------------------#
# -----------------------------------------------------------------------------#

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self,
                 from_rep='axis_angle',
                 to_rep='rotation_6d',
                 from_convention=None,
                 to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(rc, f'{from_rep}_to_matrix'),
                getattr(rc, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention)
                         for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(rc, f'matrix_to_{to_rep}'),
                getattr(rc, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention)
                         for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.tensor(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(self, x: Union[np.ndarray, torch.Tensor]
                ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]
                ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


# -----------------------------------------------------------------------------#
# --------------------------- multi-field normalizer --------------------------#
# -----------------------------------------------------------------------------#

def empirical_cdf(sample):
    """ https://stackoverflow.com/a/33346366 """

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob


class CDFNormalizer1d:
    """
        CDF normalizer for a single dimension
    """

    def __init__(self, X):
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        quantiles, cumprob = empirical_cdf(self.X)
        self.fn = interpolate.interp1d(quantiles, cumprob)
        self.inv = interpolate.interp1d(cumprob, quantiles)
        self.xmin, self.xmax = quantiles.min(), quantiles.max()
        self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)
        y = self.fn(x)
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        x = (x + 1) / 2.
        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f'''[ dataset/normalization ] Warning: out of range in unnormalize: '''
                f'''[{x.min()}, {x.max()}] | '''
                f'''x : [{self.xmin}, {self.xmax}] | '''
                f'''y: [{self.ymin}, {self.ymax}]''')
        x = np.clip(x, self.ymin, self.ymax)
        y = self.inv(x)
        return y


class CDFNormalizer:
    """
        makes training data uniform (over each dimension) by transforming it with marginal CDFs
    """

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins, self.maxs = X.min(0), X.max(0)
        self.dim = X.shape[-1]
        self.cdfs = [
            CDFNormalizer1d(self.X[:, i])
            for i in range(self.dim)]

    def wrap(self, fn_name, x):
        shape = x.shape
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap('normalize', x)

    def unnormalize(self, x):
        return self.wrap('unnormalize', x)


class GaussianNormalizer:
    """
        normalizes data to have zero mean and unit variance
    """

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.means, self.stds = X.mean(0), X.std(0)
        self.stds[self.stds == 0] = 1.

    def normalize(self, x):
        return (x - self.means[None,]) / self.stds[None,]

    def unnormalize(self, x):
        return x * self.stds[None,] + self.means[None,]


class ImageNormalizer:
    """
        normalizes image data from range [0, 1] to [-1, 1].
    """

    def __init__(self):
        pass

    def normalize(self, x):
        return x * 2.0 - 1.0

    def unnormalize(self, x):
        return (x + 1.0) / 2.0


class MinMaxNormalizer:
    """
        normalizes data through maximum and minimum expansion.
    """

    def __init__(self, X):
        X = X.reshape(-1, X.shape[-1]).astype(np.float32)
        self.min, self.max = np.min(X, axis=0), np.max(X, axis=0)
        self.range = self.max - self.min
        if np.any(self.range == 0):
            self.range = self.max - self.min
            print("Warning: Some features have the same min and max value. These will be set to 0.")
            self.range[self.range == 0] = 1

    def normalize(self, x):
        x = x.astype(np.float32)
        # nomalize to [0,1]
        nx = (x - self.min) / self.range
        # normalize to [-1, 1]
        nx = nx * 2 - 1
        return nx

    def unnormalize(self, x):
        x = x.astype(np.float32)
        nx = (x + 1) / 2
        x = nx * self.range + self.min
        return x


class EmptyNormalizer:
    """
        do nothing and change nothing
    """

    def __init__(self):
        pass

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x


# -----------------------------------------------------------------------------#
# ------------------------------- useful tool ---------------------------------#
# -----------------------------------------------------------------------------#

def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif value is None:
            result[key] = None
        else:
            result[key] = func(value)
    return result


def loop_dataloader(dl):
    while True:
        for b in dl:
            yield b

def loop_two_dataloaders(dl1, dl2):
    while True:
        for b1, b2 in zip(dl1, dl2):
            yield b1, b2