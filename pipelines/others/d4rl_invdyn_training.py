import argparse

import d4rl
import gym
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.utils import loop_dataloader
from cleandiffuser.invdynamic import FancyMlpInvDynamic

class InvDyn_Wrapper(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx], self.next_obs[idx]


if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env_name", type=str, default="halfcheetah-medium-expert-v2")
    argparser.add_argument("--device_id", type=int, default=0)
    argparser.add_argument("--hidden_dim", type=int, default=256)
    args = argparser.parse_args()
    
    env_name = args.env_name
    device_id = args.device_id
    hidden_dim = args.hidden_dim
    
    env = gym.make(env_name)
    if "kitchen" in env_name:
        dataset = InvDyn_Wrapper(D4RLKitchenTDDataset(d4rl.qlearning_dataset(env)))
    elif "antmaze" in env_name:
        dataset = InvDyn_Wrapper(D4RLAntmazeTDDataset(d4rl.qlearning_dataset(env)))
    else:
        dataset = InvDyn_Wrapper(D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env)))
        
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim
    
    invdyn = FancyMlpInvDynamic(obs_dim, act_dim, hidden_dim, add_norm=True, add_dropout=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./logs/invdyn/{env_name}/",
        filename=f"hidden_dim={hidden_dim}")
    
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)
    
    trainer = L.Trainer(
        max_steps=1000000, deterministic=True,
        accelerator="gpu", devices=[device_id,],
        default_root_dir=f"./logs/invdyn/{env_name}/",
        log_every_n_steps=1000, callbacks=[checkpoint_callback,])

    trainer.fit(invdyn, dataloader)
    
