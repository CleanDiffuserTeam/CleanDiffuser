from pathlib import Path

import d4rl
import gym
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

from cleandiffuser.dataset.d4rl_antmaze_dataset import (
    D4RLAntmazeTDDataset,
    MultiHorizonD4RLAntmazeDataset,
)
from cleandiffuser.utils import IQL, loop_dataloader, set_seed


class MultiHorizonD4RLAntmazeDatasetwQ(MultiHorizonD4RLAntmazeDataset):
    pred_values = None

    @torch.no_grad()
    def add_value(self, iql: IQL, device: str):
        self.pred_values = np.zeros_like(self.seq_rew)
        for i in tqdm(range(self.pred_values.shape[0])):
            self.pred_values[i] = iql.V(torch.tensor(self.seq_obs[i], device=device)).cpu().numpy()

    def __getitem__(self, idx: int):
        indices = [
            int(self.len_each_horizon[i] * (idx / self.len_each_horizon[-1]))
            for i in range(len(self.horizons))
        ]

        datas = []
        h, n = self.horizons, len(self.horizons)
        
        disc_tensor = 0.99 ** np.arange(h[0])[:, None]
        for i, horizon in enumerate(h):
            path_idx, start, end = self.indices[i][indices[i]]

            obs = np.copy(self.seq_obs[path_idx, start:end])
            obs = obs[::(h[i + 1] - 1) if i < n - 1 else 1]
            act = np.copy(self.seq_act[path_idx, start:end])
            act = act[::(h[i + 1] - 1) if i < n - 1 else 1]
            rew = np.copy(self.seq_rew[path_idx, start:end])
            pred_val = np.copy(self.pred_values[path_idx, start:end])
            
            rew += 1
            cum_rew = np.cumsum(rew, 0)
            mask = (cum_rew == 0.).astype(np.float32)
            old_mask = np.copy(mask)
            mask[1:] = old_mask[:-1]
            mask[0] = 1.
            val = rew.max() / mask.sum()
            if i == 0:
                val = rew - 1.
                val[-1] = pred_val[-1]
                val = (disc_tensor * val * mask).sum(0) / 100. + 1

            data = {
                "x0": np.concatenate([obs, act], axis=-1),
                "condition_cfg": val,
            }

            datas.append({"horizon": horizon, "data": data})

        return datas


# --- Config ---
env_name = "antmaze-large-play-v2"
seed = 0
mode = "iql_training"
devices = [0]
default_root_dir = Path(__file__).parents[2] / f"results/diffuserlite/{env_name}/"

if __name__ == "__main__":
    set_seed(seed)
    default_root_dir.mkdir(parents=True, exist_ok=True)

    w_cfgs = [1.0, 0.0, 0.0]
    planning_horizons = [5, 5, 9]
    n_levels = len(planning_horizons)
    temporal_horizons = [planning_horizons[-1] for _ in range(n_levels)]
    for i in range(n_levels - 1):
        temporal_horizons[-2 - i] = (planning_horizons[-2 - i] - 1) * (
            temporal_horizons[-1 - i] - 1
        ) + 1

    if mode == "iql_training":
        device = f"cuda:{devices[0]}"

        dummy_env = gym.make(env_name)
        dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(dummy_env))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True)
        obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

        iql = IQL(obs_dim, act_dim, hidden_dim=256, discount=0.99, tau=0.9)

        callback = ModelCheckpoint(
            dirpath=default_root_dir,
            filename="iql_{step}",
            every_n_train_steps=100_000,
        )
        trainer = L.Trainer(
            devices=devices,
            max_steps=1000_000,
            callbacks=[callback],
            default_root_dir=default_root_dir,
        )
        trainer.fit(iql, dataloader)

    elif mode == "training":
        device = f"cuda:{devices[0]}"
        
        dummy_env = gym.make(env_name)
        dataset = MultiHorizonD4RLAntmazeDatasetwQ(
            dummy_env.get_dataset(), horizons=temporal_horizons, discount=0.99
        )
        obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

        iql = IQL(obs_dim, act_dim, hidden_dim=512, discount=0.99, tau=0.9).to(device)
        iql.load_state_dict(torch.load(default_root_dir / "iql_step=1000000.ckpt", map_location=device)["state_dict"])
        dataset.add_value(iql, device)