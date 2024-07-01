from typing import Dict
import torch
import numpy as np
import copy
from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.dataset.dataset_utils import SequenceSampler, MinMaxNormalizer, ImageNormalizer, dict_apply


# data/pusht_cchi_v7_replay.zarr
#  ├── data
#  │   ├── action (25650, 2) float32
#  │   ├── img (25650, 96, 96, 3) float32
#  │   ├── keypoint (25650, 9, 2) float32
#  │   ├── n_contacts (25650, 1) float32
#  │   └── state (25650, 5) float32
#  └── meta
#      └── episode_ends (206,) int64
# assert('data' in root)
# assert('meta' in root)
# assert('episode_ends' in root['meta'])


class PushTStateDataset(BaseDataset):
    def __init__(self,
            dataset_path,
            obs_keys=["state", "action"],
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False
        ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, keys=obs_keys)

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
        state_normalizer = MinMaxNormalizer(self.replay_buffer['state'][:])  # (N, obs_dim)
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
        state = sample['state'].astype(np.float32)  # (T, 5)
        state = self.normalizer['obs']['state'].normalize(state)
        
        action = sample['action'].astype(np.float32)  # (T, 2)
        action = self.normalizer['action'].normalize(action)
        data = {
            'obs': {
                'state': state, # T, 5
            },
            'action': action,  # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data


class PushTKeypointDataset(BaseDataset):
    def __init__(self,
            dataset_path,
            obs_keys=['keypoint', 'state', 'action'], 
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False
        ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, keys=obs_keys)

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
        agent_pos_normalizer = MinMaxNormalizer(self.replay_buffer['state'][:, :2])  # (N, 2)
        keypoint_normalizer = MinMaxNormalizer(self.replay_buffer['keypoint'][:])  # (N, 9, 2)
        action_normalizer = MinMaxNormalizer(self.replay_buffer['action'][:])  # (N, 2)
        return {
            "obs": {
                "keypoint": keypoint_normalizer,
                "agent_pos": agent_pos_normalizer
            },
            "action": action_normalizer
        }
    
    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample): 
        # keypoint
        data_size = sample['keypoint'].shape[0]  # (T, 9, 2)
        keypoint = sample['keypoint'].reshape(-1, sample['keypoint'].shape[-1]).astype(np.float32)
        keypoint = self.normalizer['obs']['keypoint'].normalize(keypoint)
        keypoint = keypoint.reshape(data_size, -1)  # (T, 18)
        
        # agent_pos
        agent_pos = sample['state'][:,:2].astype(np.float32)  # (T, 2)
        agent_pos = self.normalizer['obs']['agent_pos'].normalize(agent_pos)
        
        # action
        action = sample['action'].astype(np.float32)  # (T, 2)
        action = self.normalizer['action'].normalize(action)
        
        data = {
            'obs': {
                'keypoint': keypoint, # T, 18
                'agent_pos': agent_pos, # T, 2
            },
            'action': action,  # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data


class PushTImageDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            obs_keys=['img', 'state', 'action'], 
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False
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
        agent_pos_normalizer = MinMaxNormalizer(self.replay_buffer['state'][...,:2])
        image_normalizer = ImageNormalizer()
        action_normalizer = MinMaxNormalizer(self.replay_buffer['action'][:])
        
        return {
            "obs": {
                "image": image_normalizer,
                "agent_pos": agent_pos_normalizer
            },
            "action": action_normalizer
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # image
        image = np.moveaxis(sample['img'], -1, 1) / 255
        image = self.normalizer['obs']['image'].normalize(image)
        
        # agent_pos
        agent_pos = sample['state'][:,:2].astype(np.float32)  # (T, 2)
        agent_pos = self.normalizer['obs']['agent_pos'].normalize(agent_pos)
        
        # action
        action = sample['action'].astype(np.float32)  # (T, 2)
        action = self.normalizer['action'].normalize(action)
        
        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': action,  # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data
