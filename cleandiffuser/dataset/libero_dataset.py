from typing import List

import numpy as np
import torch
import zarr
from cleandiffuser.utils import MinMaxNormalizer, create_indices
from cleandiffuser.utils.codecs import jpeg  # noqa

FULL_OBS_LIST = [
    "color",
    "color_ego",
    "depth",
    "depth_ego",
    "pointcloud",
    "pointcloud_ego",
    "eef_states",
    "gripper_states",
    "joint_states",
    "states",
]

ACTION_MINMAX = {
    "libero_goal": {
        "max": np.array(
            [0.9375, 0.9375, 0.9375, 0.3558, 0.375, 0.375, 1.0], dtype=np.float32
        ),
        "min": np.array(
            [-0.9375, -0.9375, -0.9375, -0.2583, -0.375, -0.2872, -1.0],
            dtype=np.float32,
        ),
    },
    "libero_spatial": {
        "max": np.array(
            [0.9375, 0.9375, 0.9375, 0.1972, 0.3365, 0.375, 1.0], dtype=np.float32
        ),
        "min": np.array(
            [-0.9375, -0.9375, -0.9375, -0.1886, -0.3675, -0.36, -1.0], dtype=np.float32
        ),
    },
    "libero_object": {
        "max": np.array(
            [0.9375, 0.90, 0.9375, 0.18, 0.375, 0.19, 1.0], dtype=np.float32
        ),
        "min": np.array(
            [-0.89, -0.9375, -0.9375, -0.15, -0.356, -0.33, -1.0], dtype=np.float32
        ),
    },
    "libero_10": {
        "max": np.array(
            [0.9375, 0.92, 0.925, 0.305, 0.313, 0.375, 1.0], dtype=np.float32
        ),
        "min": np.array(
            [-0.9375, -0.922, -0.9375, -0.24, -0.304, -0.3675, -1.0], dtype=np.float32
        ),
    },
}


class LiberoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        observation_meta: List[str] = FULL_OBS_LIST,
        To: int = 2,
        Ta: int = 16,
    ):
        super().__init__()
        if "libero_goal" in str(data_path):
            act_min, act_max = (
                ACTION_MINMAX["libero_goal"]["min"],
                ACTION_MINMAX["libero_goal"]["max"],
            )
        elif "libero_spatial" in str(data_path):
            act_min, act_max = (
                ACTION_MINMAX["libero_spatial"]["min"],
                ACTION_MINMAX["libero_spatial"]["max"],
            )
        elif "libero_object" in str(data_path):
            act_min, act_max = (
                ACTION_MINMAX["libero_object"]["min"],
                ACTION_MINMAX["libero_object"]["max"],
            )
        elif "libero_10" in str(data_path):
            act_min, act_max = (
                ACTION_MINMAX["libero_10"]["min"],
                ACTION_MINMAX["libero_10"]["max"],
            )

        self.root = zarr.open(str(data_path), mode="r")
        self._obs_meta = observation_meta
        self.To, self.Ta = To, Ta
        self._episode_ends = self.root.meta["episode_ends"][:]

        self.indices = create_indices(
            episode_ends=self._episode_ends,
            sequence_length=Ta + To - 1,
            pad_before=To - 1,
            pad_after=Ta - 1,
        )
        self.episode_idx = np.empty((len(self.indices),), dtype=int)
        for i in range(len(self.indices)):
            end_idx = self.indices[i][-1]
            self.episode_idx[i] = np.searchsorted(self._episode_ends, end_idx)

        self.size = self.root.data.actions.shape[0]

        self.normalizers = {
            "action": MinMaxNormalizer(X_min=act_min, X_max=act_max),
        }

    def get_normalizer(self):
        return self.normalizers

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            end_idx,
        ) = self.indices[idx]
        episode_idx = self.episode_idx[idx]

        e_Ta = self.Ta - (self.To + self.Ta - 1 - sample_end_idx)
        action = self.root.data.actions[buffer_end_idx - e_Ta : buffer_end_idx]
        if self.To + self.Ta - 1 > sample_end_idx:
            action = np.pad(
                action,
                ((0, self.To + self.Ta - 1 - sample_end_idx), (0, 0)),
                mode="edge",
            )
        assert action.shape[0] == self.Ta, f"{action.shape[0]} != {self.Ta}"
        action = self.normalizers["action"].normalize(action)

        observation = dict()
        for obs_name in self._obs_meta:
            x = self.root.data[obs_name][
                buffer_start_idx : buffer_start_idx + self.To - sample_start_idx
            ]
            if sample_start_idx > 0:
                if "color" in obs_name:
                    x = np.pad(x, ((sample_start_idx, 0), *[(0, 0)] * 3), mode="edge")

                elif "depth" in obs_name:
                    # from mm to m, reshape to (C, H, W)
                    x = (x.astype(np.float32) / 1000.0)[None]
                    x = np.pad(x, ((sample_start_idx, 0), *[(0, 0)] * 3), mode="edge")

                elif "pointcloud" in obs_name:
                    x = np.pad(x, ((sample_start_idx, 0), *[(0, 0)] * 2), mode="edge")

                else:
                    x = np.pad(x, ((sample_start_idx, 0), (0, 0)), mode="edge")

            observation[obs_name] = x

        return {
            "observation": observation,
            "action": action,
            "language_embedding": self.root.meta.language_embeddings[episode_idx],
            "language_mask": self.root.meta.language_masks[episode_idx],
        }
