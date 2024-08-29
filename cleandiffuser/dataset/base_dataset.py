from typing import Dict

import torch.nn
from torch.utils.data import Dataset


# Adapted from the datasets on: https://github.com/real-stanford/diffusion_policy

# Observation Horizon: To|n_obs_steps
# Action Horizon: Ta|n_action_steps
# Prediction Horizon: T|horizon
# To = 3
# Ta = 4
# T = 6
# |o|o|o|
# | | |a|a|a|a|
# To = 2
# Ta = 5
# T = 6
# |o|o|
# | |a|a|a|a|a|


class BaseDataset(Dataset):
    def get_normalizer(self, **kwargs):
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key1: T, Do1  # default key: state
                key2: T, Do2
            action: T, Da
            reward: T, 1
            info: 
        """
        raise NotImplementedError()
