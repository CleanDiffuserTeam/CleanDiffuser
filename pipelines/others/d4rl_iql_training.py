import argparse

import d4rl
import gym
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.utils import loop_dataloader
from cleandiffuser.utils.iql_L import IQL


class IQLD4RLMuJoCoTDDataset(D4RLMuJoCoTDDataset):
    def __getitem__(self, idx: int):
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.tml[idx]
    

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env_name", type=str, default="halfcheetah-medium-expert-v2")
    argparser.add_argument("--device_id", type=int, default=0)
    args = argparser.parse_args()
    
    env_name = args.env_name
    device_id = args.device_id
    normalize_reward = False
    tau = 0.7
    discount = 0.99
    hidden_dim = 256
    
    env = gym.make(env_name)
    dataset = IQLD4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), normalize_reward=normalize_reward)
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim
    
    iql = IQL(obs_dim, act_dim, tau, discount, hidden_dim)
    
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)
    
    trainer = L.Trainer(
        max_steps=1000000, deterministic=True, 
        accelerator="gpu", devices=[device_id,], default_root_dir=f"./logs/iql/{env_name}/",
        log_every_n_steps=1000)
    
    trainer.fit(iql, loop_dataloader(dataloader))
    