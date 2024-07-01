import numpy as np
import torch

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.dataset_utils import GaussianNormalizer, dict_apply


class D4RLKitchenDataset(BaseDataset):
    """
    In D4RL Kitchen, `terminal` means a demonstration is finished, and we need penalty.
    Padding to repeat the last state-action-reward until the end of the sequence.
    """
    def __init__(
            self,
            dataset,
            horizon=1,
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

        self.horizon = horizon
        self.o_dim, self.a_dim = observations.shape[-1], actions.shape[-1]
        self.discount = discount ** np.arange(max_path_length, dtype=np.float32)

        self.indices = []
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
                _seq_act[i - ptr + 1:] = 0                       # repeat zero action
                _seq_rew[i - ptr + 1:] = rewards[i]              # repeat last reward
                
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

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]

        rewards = self.seq_rew[path_idx, start:]
        values = (rewards * self.discount[:rewards.shape[0], None]).sum(0)

        data = {
            'obs': {
                'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
            'rew': self.seq_rew[path_idx, start:end],
            'val': values}

        torch_data = dict_apply(data, torch.tensor)

        return torch_data


class D4RLKitchenTDDataset(BaseDataset):
    def __init__(self, dataset):
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
