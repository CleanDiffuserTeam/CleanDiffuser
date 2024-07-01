from typing import Dict
import torch
import numpy as np
import copy
import pathlib
from tqdm import tqdm
from cleandiffuser.env.kitchen.kitchen_util import parse_mjl_logs
from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.dataset.dataset_utils import SequenceSampler, MinMaxNormalizer, dict_apply


class KitchenDataset(BaseDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
        ):
        super().__init__()

        data_directory = pathlib.Path(dataset_dir)
        observations = np.load(data_directory / "observations_seq.npy")
        actions = np.load(data_directory / "actions_seq.npy")
        masks = np.load(data_directory / "existence_mask.npy")

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i in range(len(masks)):
            eps_len = int(masks[i].sum())
            obs = observations[i,:eps_len].astype(np.float32)
            action = actions[i,:eps_len].astype(np.float32)
            data = {                              
                'state': obs,
                'action': action
            }
            self.replay_buffer.add_episode(data)
        
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

    def sample_to_data(self, sample):
        state = sample['state'].astype(np.float32) 
        state = self.normalizer['obs']['state'].normalize(state)
        
        action = sample['action'].astype(np.float32)  
        action = self.normalizer['action'].normalize(action)
        data = {
            'obs': {
                'state': state,
            },
            'action': action, 
        }
        return data
        
    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data
    

class KitchenMjlDataset(BaseDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=True,
            robot_noise_ratio=0.1,
        ):
        super().__init__()

        data_directory = pathlib.Path(dataset_dir)
        robot_pos_noise_amp = np.array([0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   ,
            0.1   , 0.005 , 0.005 , 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
            0.0005, 0.005 , 0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ,
            0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ], dtype=np.float32)
        rng = np.random.default_rng(seed=42)

        data_directory = pathlib.Path(dataset_dir)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i, mjl_path in enumerate(tqdm(list(data_directory.glob('*/*.mjl')))):
            try:
                data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                qpos = data['qpos'].astype(np.float32)
                obs = np.concatenate([
                    qpos[:,:9],
                    qpos[:,-21:],
                    np.zeros((len(qpos),30),dtype=np.float32)
                ], axis=-1)
                if robot_noise_ratio > 0:
                    # add observation noise to match real robot
                    noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                        low=-1., high=1., size=(obs.shape[0], 30))
                    obs[:,:30] += noise
                episode = {
                    'state': obs,
                    'action': data['ctrl'].astype(np.float32)
                }
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                print(i, e)
        
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

    def sample_to_data(self, sample):
        state = sample['state'].astype(np.float32) 
        state = self.normalizer['obs']['state'].normalize(state)
        
        action = sample['action'].astype(np.float32)  
        action = self.normalizer['action'].normalize(action)
        data = {
            'obs': {
                'state': state,
            },
            'action': action, 
        }
        return data
        
    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data