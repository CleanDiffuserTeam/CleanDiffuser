from typing import Dict
import torch
import numpy as np
import copy
import pathlib
from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.dataset.dataset_utils import SequenceSampler, MinMaxNormalizer, EmptyNormalizer, dict_apply

#  dev/block_pushing/multimodal_push_seed.zarr
#  ├── data
#  │   ├── action (114962, 2) float32
#  │   └── obs (114962, 16) float32
#  └── meta
#      └── episode_ends (1000,) int64

class BlockPushDataset(BaseDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_keys=['obs', 'action']
        ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=obs_keys)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after)

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        self.normalizer = self.get_normalizer()
    
    def get_normalizer(self):
        state_normalizer = MinMaxNormalizer(self.replay_buffer['obs'][:])  # (N, obs_dim)
        action_normalizer = MinMaxNormalizer(self.replay_buffer['action'][:])  # (N, action_dim)

        return {
            "obs": {
                "state": state_normalizer
            },
            "action": action_normalizer
        }
        
    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        state = sample['obs'].astype(np.float32)  # (T, 5)
        state = self.normalizer['obs']['state'].normalize(state)
        
        action = sample['action'].astype(np.float32)  # (T, 2)
        action = self.normalizer['action'].normalize(action)
        data = {
            'obs': {
                'state': state, # T, D_o    
            },
            'action': action, # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data