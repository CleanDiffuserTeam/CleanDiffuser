import os
import pathlib

import h5py
import numpy as np
from tqdm import tqdm

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.dataset_utils import SequenceSampler
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.env.kitchen.kitchen_util import parse_mjl_logs
from cleandiffuser.utils import MinMaxNormalizer


class KitchenDataset(BaseDataset):
    """Relay Kitchen imitation learning dataset.

    The dataset chunks the demonstrations into sequences of length `horizon`.
    It uses `MinMaxNormalizer` to normalize the observations and actions to [-1, 1] as default.
    Each batch contains:
    - batch['state'], observation of shape (batch_size, horizon, obs_dim)
    - batch['action'], action of shape (batch_size, horizon, act_dim)

    Args:
        dataset_dir (str):
            Path to the dataset directory. Please download from https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip and unzip it.
        horizon (int):
            The length of the sequence.
        pad_before (int):
            The number of steps to pad the beginning of the sequence.
        pad_after (int):
            The number of steps to pad the end of the sequence.
        abs_action (bool):
            Whether to use absolute action (Position control).
        robot_noise_ratio (float):
            The ratio of robot position noise. Only used when `abs_action` is True.

    Examples:
        >>> dataset = KitchenDataset(dataset_dir='dev/kitchen', horizon=4)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> batch = next(iter(dataloader))
        >>> obs = batch["state"]  # (32, 4, 60)
        >>> act = batch["action"]  # (32, 4, 9)

        >>> normalizer = dataset.get_normalizer()
        >>> obs = env.reset()[None, :]  # (1, 60)
        >>> obs = normalizer["state"].normalize(obs)
        >>> action = behavior_clone_policy(obs)  # (1, 9)
        >>> action = normalizer["action"].unnormalize(action)
        >>> obs, rew, done, info = env.step(action)
    """

    def __init__(
        self,
        dataset_dir: str = "dev/kitchen",
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        abs_action: bool = True,
        robot_noise_ratio: float = 0.1,
    ):
        super().__init__()

        if abs_action:
            data_directory = pathlib.Path(dataset_dir)

            if not os.path.exists(data_directory / "kitchen_abs_action_dataset.hdf5"):
                print("No abs action dataset found. Generating...")
                robot_pos_noise_amp = np.array(
                    [0.1] * 9
                    + [0.005] * 2
                    + [0.0005] * 6
                    + [0.005] * 3
                    + [0.1] * 3
                    + [0.005] * 3
                    + [0.1] * 3
                    + [0.005],
                    dtype=np.float32,
                )
                rng = np.random.default_rng(seed=42)

                self.replay_buffer = ReplayBuffer.create_empty_numpy()

                for i, mjl_path in enumerate(
                    tqdm(list((data_directory / "kitchen_demos_multitask").glob("*/*.mjl")))
                ):
                    try:
                        data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                        qpos = data["qpos"].astype(np.float32)
                        obs = np.concatenate(
                            [
                                qpos[:, :9],
                                qpos[:, -21:],
                                np.zeros((len(qpos), 30), dtype=np.float32),
                            ],
                            axis=-1,
                        )
                        if robot_noise_ratio > 0:
                            # add observation noise to match real robot
                            noise = (
                                robot_noise_ratio
                                * robot_pos_noise_amp
                                * rng.uniform(low=-1.0, high=1.0, size=(obs.shape[0], 30))
                            )
                            obs[:, :30] += noise
                        episode = {"state": obs, "action": data["ctrl"].astype(np.float32)}
                        self.replay_buffer.add_episode(episode)
                    except Exception as e:
                        print(i, e)

                self.state_normalizer = MinMaxNormalizer(self.replay_buffer.root["data"]["state"])
                self.action_normalizer = MinMaxNormalizer(self.replay_buffer.root["data"]["action"])

                normalized_states = self.state_normalizer.normalize(
                    self.replay_buffer.root["data"]["state"]
                )
                normalized_actions = self.action_normalizer.normalize(
                    self.replay_buffer.root["data"]["action"]
                )

                with h5py.File(data_directory / "kitchen_abs_action_dataset.hdf5", "w") as f:
                    f.create_dataset("state", data=normalized_states)
                    f.create_dataset("action", data=normalized_actions)
                    f.create_dataset(
                        "episode_ends", data=self.replay_buffer.root["meta"]["episode_ends"]
                    )
                    f.create_dataset("state_normalizer_min", data=self.state_normalizer.min)
                    f.create_dataset("state_normalizer_max", data=self.state_normalizer.max)
                    f.create_dataset("action_normalizer_min", data=self.action_normalizer.min)
                    f.create_dataset("action_normalizer_max", data=self.action_normalizer.max)

            else:
                print("Abs action dataset found. Loading...")

            with h5py.File(data_directory / "kitchen_abs_action_dataset.hdf5", "r") as f:
                dataset = {
                    "data": {
                        "state": np.array(f["state"], dtype=np.float32),
                        "action": np.array(f["action"], dtype=np.float32),
                    },
                    "meta": {"episode_ends": np.array(f["episode_ends"], dtype=np.int64)},
                }
                self.replay_buffer = ReplayBuffer.create_from_dict(root=dataset)
                self.state_normalizer = MinMaxNormalizer(
                    X_min=np.array(f["state_normalizer_min"], dtype=np.float32),
                    X_max=np.array(f["state_normalizer_max"], dtype=np.float32),
                )
                self.action_normalizer = MinMaxNormalizer(
                    X_min=np.array(f["action_normalizer_min"], dtype=np.float32),
                    X_max=np.array(f["action_normalizer_max"], dtype=np.float32),
                )

        else:
            data_directory = pathlib.Path(dataset_dir)
            observations = np.load(data_directory / "observations_seq.npy").astype(np.float32)
            actions = np.load(data_directory / "actions_seq.npy").astype(np.float32)
            masks = np.load(data_directory / "existence_mask.npy").astype(np.int64)

            self.state_normalizer = MinMaxNormalizer(observations)
            self.action_normalizer = MinMaxNormalizer(actions)

            normed_observations = self.state_normalizer.normalize(observations)
            normed_actions = self.action_normalizer.normalize(actions)

            self.replay_buffer = ReplayBuffer.create_empty_numpy()
            for i in range(len(masks)):
                eps_len = int(masks[i].sum())
                obs = normed_observations[i, :eps_len]
                action = normed_actions[i, :eps_len]
                data = {"state": obs, "action": action}
                self.replay_buffer.add_episode(data)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.normalizer = self.get_normalizer()
        self.obs_dim, self.act_dim = (
            self.replay_buffer["state"].shape[-1],
            self.replay_buffer["action"].shape[-1],
        )

    def get_normalizer(self):
        return {"state": self.state_normalizer, "action": self.action_normalizer}

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int):
        return self.sampler.sample_sequence(idx)
