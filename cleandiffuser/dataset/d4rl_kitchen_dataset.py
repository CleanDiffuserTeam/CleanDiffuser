from typing import Dict

import numpy as np
import torch

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.utils import GaussianNormalizer, dict_apply


class D4RLKitchenDataset(BaseDataset):
    """ **D4RL-Kitchen Sequential Dataset**

    torch.utils.data.Dataset wrapper for D4RL-Kitchen dataset.
    Chunk the dataset into sequences of length `horizon` with obs-repeat/act-zero/reward-repeat padding.
    Use GaussianNormalizer to normalize the observations as default.
    Each batch contains
    - batch["obs"]["state"], observations of shape (batch_size, horizon, o_dim)
    - batch["act"], actions of shape (batch_size, horizon, a_dim)
    - batch["rew"], rewards of shape (batch_size, horizon, 1)
    - batch["val"], Monte Carlo return of shape (batch_size, 1)

    Args:
        dataset: Dict[str, np.ndarray],
            D4RL-Kitchen dataset. Obtained by calling `env.get_dataset()`.
        horizon: int,
            Length of each sequence. Default is 1.
        max_path_length: int,
            Maximum length of the episodes. Default is 280.
        discount: float,
            Discount factor. Default is 0.99.

    Examples:
        >>> env = gym.make("kitchen-mixed-v0")
        >>> dataset = D4RLKitchenDataset(env.get_dataset(), horizon=32)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> batch = next(iter(dataloader))
        >>> obs = batch["obs"]["state"]  # (32, 32, 60)
        >>> act = batch["act"]           # (32, 32, 9)
        >>> rew = batch["rew"]           # (32, 32, 1)
        >>> val = batch["val"]           # (32, 1)

        >>> normalizer = dataset.get_normalizer()
        >>> obs = env.reset()[None, :]
        >>> normed_obs = normalizer.normalize(obs)
        >>> unnormed_obs = normalizer.unnormalize(normed_obs)
    """
    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            horizon: int = 1,
            max_path_length: int = 280,
            discount: float = 0.99,
    ):
        super().__init__()

        observations, actions, rewards, timeouts, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["timeouts"],
            dataset["terminals"])
        self.normalizers = {
            "state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        self.indices = []
        self.seq_obs, self.seq_act, self.seq_rew = [], [], []
        self.tml_and_not_timeout = []

        self.path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i] or i == timeouts.shape[0] - 1:
                self.path_lengths.append(i - ptr + 1)

                if terminals[i] and not timeouts[i]:
                    self.tml_and_not_timeout.append([path_idx, i - ptr])

                _seq_obs = np.zeros((max_path_length, self.o_dim), dtype=np.float32)
                _seq_act = np.zeros((max_path_length, self.a_dim), dtype=np.float32)
                _seq_rew = np.zeros((max_path_length, 1), dtype=np.float32)

                _seq_obs[:i - ptr + 1] = normed_observations[ptr:i + 1]
                _seq_act[:i - ptr + 1] = actions[ptr:i + 1]
                _seq_rew[:i - ptr + 1] = rewards[ptr:i + 1][:, None]

                # repeat padding
                _seq_obs[i - ptr + 1:] = normed_observations[i]  # repeat last state
                _seq_act[i - ptr + 1:] = 0  # repeat zero action
                _seq_rew[i - ptr + 1:] = rewards[i]  # repeat last reward

                self.seq_obs.append(_seq_obs)
                self.seq_act.append(_seq_act)
                self.seq_rew.append(_seq_rew)

                max_start = min(self.path_lengths[-1] - 1, max_path_length - horizon)
                self.indices += [(path_idx, start, start + horizon) for start in range(max_start + 1)]

                ptr = i + 1
                path_idx += 1

        self.seq_obs = np.array(self.seq_obs)
        self.seq_act = np.array(self.seq_act)
        self.seq_rew = np.array(self.seq_rew)

        self.seq_val = np.copy(self.seq_rew)
        for i in range(max_path_length - 1):
            self.seq_val[:, - 2 - i] = self.seq_rew[:, -2 - i] + discount * self.seq_val[:, -1 - i]
        self.tml_and_not_timeout = np.array(self.tml_and_not_timeout, dtype=np.int64)

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]

        data = {
            'obs': {
                'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
            'rew': self.seq_rew[path_idx, start:end],
            'val': self.seq_val[path_idx, start],
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data


class D4RLKitchenTDDataset(BaseDataset):
    """ **D4RL-Kitchen Transition Dataset**

    torch.utils.data.Dataset wrapper for D4RL-Kitchen dataset.
    Chunk the dataset into transitions.
    Use GaussianNormalizer to normalize the observations as default.
    Each batch contains
    - batch["obs"]["state"], observation of shape (batch_size, o_dim)
    - batch["next_obs"]["state"], next observation of shape (batch_size, o_dim)
    - batch["act"], action of shape (batch_size, a_dim)
    - batch["rew"], reward of shape (batch_size, 1)
    - batch["tml"], terminal of shape (batch_size, 1)

    Args:
        dataset: Dict[str, np.ndarray],
            D4RL-MuJoCo TD dataset. Obtained by calling `d4rl.qlearning_dataset(env)`.

    Examples:
        >>> env = gym.make("kitchen-mixed-v0")
        >>> dataset = D4RLKitchenTDDataset(d4rl.qlearning_dataset(env))
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> batch = next(iter(dataloader))
        >>> obs = batch["obs"]["state"]  # (32, 60)
        >>> act = batch["act"]           # (32, 9)
        >>> rew = batch["rew"]           # (32, 1)
        >>> tml = batch["tml"]           # (32, 1)
        >>> next_obs = batch["next_obs"]["state"]  # (32, 60)

        >>> normalizer = dataset.get_normalizer()
        >>> obs = env.reset()[None, :]
        >>> normed_obs = normalizer.normalize(obs)
        >>> unnormed_obs = normalizer.unnormalize(normed_obs)
    """
    def __init__(self, dataset: Dict[str, np.ndarray]):
        super().__init__()

        observations, actions, next_observations, rewards, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["next_observations"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["terminals"].astype(np.float32))

        self.normalizers = {
            "state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)
        normed_next_observations = self.normalizers["state"].normalize(next_observations)

        self.obs = torch.tensor(normed_observations)
        self.act = torch.tensor(actions)
        self.rew = torch.tensor(rewards)[:, None]
        self.tml = torch.tensor(terminals)[:, None]
        self.next_obs = torch.tensor(normed_next_observations)

        self.size = self.obs.shape[0]
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        data = {
            'obs': {
                'state': self.obs[idx], },
            'next_obs': {
                'state': self.next_obs[idx], },
            'act': self.act[idx],
            'rew': self.rew[idx],
            'tml': self.tml[idx], }

        return data


class MultiHorizonD4RLKitchenDataset(BaseDataset):
    def __init__(
            self,
            dataset,
            horizons=(10, 20),
            max_path_length=280,
            discount=0.99,
    ):
        super().__init__()

        observations, actions, rewards, timeouts, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["timeouts"],
            dataset["terminals"])
        self.normalizers = {
            "state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizons = horizons
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]
        self.discount = discount ** np.arange(max_path_length, dtype=np.float32)

        self.indices = [[] for _ in range(len(horizons))]
        self.seq_obs, self.seq_act, self.seq_rew = [], [], []

        self.path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i] or i == timeouts.shape[0] - 1:
                self.path_lengths.append(i - ptr + 1)

                _seq_obs = np.zeros((max_path_length, self.o_dim), dtype=np.float32)
                _seq_act = np.zeros((max_path_length, self.a_dim), dtype=np.float32)
                _seq_rew = np.zeros((max_path_length, 1), dtype=np.float32)

                _seq_obs[:i - ptr + 1] = normed_observations[ptr:i + 1]
                _seq_act[:i - ptr + 1] = actions[ptr:i + 1]
                _seq_rew[:i - ptr + 1] = rewards[ptr:i + 1][:, None]

                # repeat padding
                _seq_obs[i - ptr + 1:] = normed_observations[i]  # repeat last state
                _seq_act[i - ptr + 1:] = 0  # repeat zero action
                _seq_rew[i - ptr + 1:] = rewards[i]  # repeat last reward

                self.seq_obs.append(_seq_obs)
                self.seq_act.append(_seq_act)
                self.seq_rew.append(_seq_rew)

                max_starts = [min(self.path_lengths[-1] - 1, max_path_length - horizon) for horizon in horizons]
                for k in range(len(horizons)):
                    self.indices[k] += [(path_idx, start, start + horizons[k]) for start in range(max_starts[k] + 1)]

                ptr = i + 1
                path_idx += 1

        self.seq_obs = np.array(self.seq_obs)
        self.seq_act = np.array(self.seq_act)
        self.seq_rew = np.array(self.seq_rew)

        self.len_each_horizon = [len(indices) for indices in self.indices]

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return max(self.len_each_horizon)

    def __getitem__(self, idx: int):

        indices = [
            int(self.len_each_horizon[i] * (idx / self.len_each_horizon[-1])) for i in range(len(self.horizons))]

        torch_datas = []

        for i, horizon in enumerate(self.horizons):
            path_idx, start, end = self.indices[i][indices[i]]

            rewards = self.seq_rew[path_idx, start:]
            values = (rewards * self.discount[:rewards.shape[0], None]).sum(0)

            data = {
                'obs': {
                    'state': self.seq_obs[path_idx, start:end]},
                'act': self.seq_act[path_idx, start:end],
                'rew': self.seq_rew[path_idx, start:end],
                'val': values}

            torch_data = dict_apply(data, torch.tensor)

            torch_datas.append({
                "horizon": horizon,
                "data": torch_data,
            })

        return torch_datas

class DV_D4RLKitchenSeqDataset(BaseDataset):
    """ **D4RL-Kitchen Sequential Dataset**

    torch.utils.data.Dataset wrapper for D4RL-Kitchen dataset.
    Chunk the dataset into sequences of length `horizon` with obs-repeat/act-zero/reward-repeat padding.
    Use GaussianNormalizer to normalize the observations as default.
    Each batch contains
    - batch["obs"]["state"], observations of shape (batch_size, horizon, o_dim)
    - batch["act"], actions of shape (batch_size, horizon, a_dim)
    - batch["rew"], rewards of shape (batch_size, horizon, 1)
    - batch["val"], Monte Carlo return of shape (batch_size, 1)

    Args:
        dataset: Dict[str, np.ndarray],
            D4RL-Kitchen dataset. Obtained by calling `env.get_dataset()`.
        horizon: int,
            Length of each sequence. Default is 1.
        max_path_length: int,
            Maximum length of the episodes. Default is 280.
        discount: float,
            Discount factor. Default is 0.99.

    Examples:
        >>> env = gym.make("kitchen-mixed-v0")
        >>> dataset = D4RLKitchenDataset(env.get_dataset(), horizon=32)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> batch = next(iter(dataloader))
        >>> obs = batch["obs"]["state"]  # (32, 32, 60)
        >>> act = batch["act"]           # (32, 32, 9)
        >>> rew = batch["rew"]           # (32, 32, 1)
        >>> val = batch["val"]           # (32, 1)

        >>> normalizer = dataset.get_normalizer()
        >>> obs = env.reset()[None, :]
        >>> normed_obs = normalizer.normalize(obs)
        >>> unnormed_obs = normalizer.unnormalize(normed_obs)
    """
    def __init__(
            self,
            dataset: Dict[str, np.ndarray],
            horizon: int = 1,
            max_path_length: int = 280,
            discount: float = 0.99,
            center_mapping: bool = True,
            stride: int = 1,
    ):
        super().__init__()

        observations, actions, rewards, timeouts, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["timeouts"].astype(np.float32),
            dataset["terminals"].astype(np.float32))
        self.stride = stride

        self.normalizers = {
            "state": GaussianNormalizer(observations)}
        normed_observations = self.normalizers["state"].normalize(observations)

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]

        self.indices = []
        self.seq_obs, self.seq_act, self.seq_rew = [], [], []

        ptr = 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i] or i == timeouts.shape[0] - 1:
                path_length = i - ptr + 1
                assert path_length <= max_path_length

                _seq_obs = np.zeros((max_path_length + (horizon - 1) * stride, self.o_dim), dtype=np.float32)
                _seq_act = np.zeros((max_path_length + (horizon - 1) * stride, self.a_dim), dtype=np.float32)
                _seq_rew = np.zeros((max_path_length + (horizon - 1) * stride, 1), dtype=np.float32)

                _seq_obs[:path_length] = normed_observations[ptr:i + 1]
                _seq_act[:path_length] = actions[ptr:i + 1]
                _seq_rew[:path_length] = rewards[ptr:i + 1][:, None]

                # repeat padding
                _seq_obs[path_length:] = normed_observations[i]  # repeat last state
                _seq_act[path_length:] = 0
                _seq_rew[path_length:] = rewards[i]

                self.seq_obs.append(_seq_obs)
                self.seq_act.append(_seq_act)
                self.seq_rew.append(_seq_rew)

                max_start = path_length - 1
                self.indices += [(path_idx, start, start + (horizon - 1) * stride + 1) for start in range(max_start + 1)]

                ptr = i + 1
                path_idx += 1

        self.seq_obs = np.array(self.seq_obs)
        self.seq_act = np.array(self.seq_act)
        self.seq_rew = np.array(self.seq_rew)

        self.seq_val = np.copy(self.seq_rew)
        for i in reversed(range(max_path_length - 1)):
            self.seq_val[:, i] = self.seq_rew[:, i] + discount * self.seq_val[:, i+1]
        
        print(f"max discounted return: {self.seq_val.max()}")
        print(f"min discounted return: {self.seq_val.min()}")
        
        # val \in [-1, 1]
        self.seq_val = (self.seq_val - self.seq_val.min()) / (self.seq_val.max() - self.seq_val.min())
        if center_mapping:
            self.seq_val = self.seq_val * 2 - 1
        print(f"max normed discounted return: {self.seq_val.max()}")
        print(f"min normed discounted return: {self.seq_val.min()}")

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]
        
        horizon_state = self.seq_obs[path_idx, start:end:self.stride]

        data = {
            'obs': {'state': horizon_state},
            'act': self.seq_act[path_idx, start:end:self.stride],
            'rew': self.seq_rew[path_idx, start:end:self.stride],
            'val': self.seq_val[path_idx, start],
        }

        torch_data = dict_apply(data, torch.tensor)

        return torch_data