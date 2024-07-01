import numpy as np
import torch

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.dataset_utils import GaussianNormalizer, dict_apply


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, max_episode_steps=1000):
    min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
    dataset["rewards"] /= max_ret - min_ret
    dataset["rewards"] *= max_episode_steps
    return dataset


class D4RLMuJoCoDataset(BaseDataset):
    def __init__(
            self,
            dataset,
            terminal_penalty=-100,
            horizon=1,
            max_path_length=1000,
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

        n_paths = np.sum(np.logical_or(terminals, timeouts))
        self.seq_obs = np.zeros((n_paths, max_path_length, self.o_dim), dtype=np.float32)
        self.seq_act = np.zeros((n_paths, max_path_length, self.a_dim), dtype=np.float32)
        self.seq_rew = np.zeros((n_paths, max_path_length, 1), dtype=np.float32)
        self.indices = []

        path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i]:
                path_lengths.append(i - ptr + 1)

                if terminals[i] and not timeouts[i]:
                    rewards[i] = terminal_penalty if terminal_penalty is not None else rewards[i]

                self.seq_obs[path_idx, :i - ptr + 1] = normed_observations[ptr:i + 1]
                self.seq_act[path_idx, :i - ptr + 1] = actions[ptr:i + 1]
                self.seq_rew[path_idx, :i - ptr + 1] = rewards[ptr:i + 1][:, None]

                max_start = min(path_lengths[-1] - 1, max_path_length - horizon)
                self.indices += [(path_idx, start, start + horizon) for start in range(max_start + 1)]

                ptr = i + 1
                path_idx += 1

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
            'val': values}

        torch_data = dict_apply(data, torch.tensor)

        return torch_data


def cosine_similarity(tensor1, tensor2):
    norm_tensor1 = torch.norm(tensor1, dim=-1, keepdim=True)
    norm_tensor2 = torch.norm(tensor2, dim=-1, keepdim=True)

    normalized_tensor1 = tensor1 / norm_tensor1
    normalized_tensor2 = tensor2 / norm_tensor2

    dot_product = torch.einsum('ntd, mtd -> nm', normalized_tensor1, normalized_tensor2)
    return dot_product


class D4RLMuJoCoRAGDataset(D4RLMuJoCoDataset):
    def __init__(
            self,
            dataset,
            terminal_penalty=-100,
            horizon=1,
            max_path_length=1000,
            discount=0.99,
    ):
        super().__init__(dataset, terminal_penalty, horizon, max_path_length, discount)

        print(f"Building dataset RAG...")
        self.returnDB = []
        self.observationsDB = []
        self.statesDB = []

        N = len(self.indices)
        num_samples = N // 20
        random_indices = torch.randperm(N)[:num_samples]
        for idx in range(len(self.indices)):
            p_idx, start, end = self.indices[idx]
            rewards = self.seq_rew[p_idx, start:]
            values = (rewards * self.discount[:rewards.shape[0], None]).sum(0)
            state = self.seq_obs[p_idx, start:end]
            self.statesDB.append(state[0])
            self.returnDB.append(values)
            self.observationsDB.append(state)
        self.statesDB = torch.tensor(np.array(self.statesDB))[random_indices]
        self.returnDB = torch.tensor(np.array(self.returnDB))[random_indices].squeeze(1)
        self.observationsDB = torch.tensor(np.array(self.observationsDB))[random_indices]
        print(f"RAG built.")

    def device_transfer(self, tensor):
        if tensor.device != self.statesDB.device:
            self.statesDB = self.statesDB.to(tensor.device)
            self.returnDB = self.returnDB.to(tensor.device)
            self.observationsDB = self.observationsDB.to(tensor.device)

    def query_state(self, state_condition):
        self.device_transfer(state_condition)
        DB = self.statesDB
        dot_products = torch.matmul(state_condition, DB.transpose(1, 0))
        norm_query = torch.norm(state_condition, dim=1).unsqueeze(1)
        norm_other_vectors = torch.norm(DB, dim=1).unsqueeze(0)
        similarities = dot_products / torch.matmul(norm_query, norm_other_vectors)
        sorted_indices = torch.argsort(-similarities, dim=1)
        top_20_indices = sorted_indices[:, :20]
        return top_20_indices

    def query_observation(self, path_condition):
        self.device_transfer(path_condition)
        similarities = cosine_similarity(path_condition, self.observationsDB)
        sorted_indices = torch.argsort(-similarities, dim=1)
        top_20_indices = sorted_indices[:, :20]
        return top_20_indices

    def find_nearest_score(self, indices, reference_score):
        b = indices.shape[0]
        selected_scores = self.returnDB[indices]

        score_diff = torch.abs(selected_scores - reference_score.unsqueeze(1))

        nearest_index = torch.argsort(score_diff, dim=1)[:, :6]
        return indices[torch.arange(b).unsqueeze(1), nearest_index]

    def find_highest_score(self, indices):
        b = indices.shape[0]
        selected_scores = self.returnDB[indices]
        highest_index = torch.argsort(-selected_scores, dim=1)[:, :5]
        return indices[torch.arange(b).unsqueeze(1), highest_index]


class D4RLMuJoCoTDDataset(BaseDataset):
    def __init__(self, dataset, normalize_reward: bool = False):
        super().__init__()
        if normalize_reward:
            dataset = modify_reward(dataset, 1000)

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


class MultiHorizonD4RLMuJoCoDataset(BaseDataset):
    def __init__(
            self,
            dataset,
            terminal_penalty=-100,
            horizons=(10, 20),
            max_path_length=1000,
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

        n_paths = np.sum(np.logical_or(terminals, timeouts))
        self.seq_obs = np.zeros((n_paths, max_path_length, self.o_dim), dtype=np.float32)
        self.seq_act = np.zeros((n_paths, max_path_length, self.a_dim), dtype=np.float32)
        self.seq_rew = np.zeros((n_paths, max_path_length, 1), dtype=np.float32)
        self.indices = [[] for _ in range(len(horizons))]

        path_lengths, ptr = [], 0
        path_idx = 0
        for i in range(timeouts.shape[0]):
            if timeouts[i] or terminals[i]:
                path_lengths.append(i - ptr + 1)

                if terminals[i] and not timeouts[i]:
                    rewards[i] = terminal_penalty if terminal_penalty is not None else rewards[i]

                self.seq_obs[path_idx, :i - ptr + 1] = normed_observations[ptr:i + 1]
                self.seq_act[path_idx, :i - ptr + 1] = actions[ptr:i + 1]
                self.seq_rew[path_idx, :i - ptr + 1] = rewards[ptr:i + 1][:, None]

                max_starts = [min(path_lengths[-1] - 1, max_path_length - horizon) for horizon in horizons]
                for k in range(len(horizons)):
                    self.indices[k] += [(path_idx, start, start + horizons[k]) for start in range(max_starts[k] + 1)]

                ptr = i + 1
                path_idx += 1

        self.len_each_horizon = [len(indices) for indices in self.indices]

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return max(self.len_each_horizon)

    def __getitem__(self, idx: int):

        indices = [np.random.randint(self.len_each_horizon[i]) for i in range(len(self.len_each_horizon))]

        torch_datas = []

        for i, horizon in enumerate(self.horizons):

            path_idx, start, end = self.indices[i][indices[i]]

            rewards = self.seq_rew[path_idx, start:]
            values = (rewards * self.discount[:rewards.shape[0], None]).sum(0)

            data = {
                'obs': {
                    'state': self.seq_obs[path_idx, start:end]},
                'act': self.seq_act[path_idx, start:end],
                'val': values}

            torch_data = dict_apply(data, torch.tensor)

            torch_datas.append({
                "horizon": horizon,
                "data": torch_data,
            })

        return torch_datas
