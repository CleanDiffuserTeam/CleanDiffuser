import os
from pathlib import Path
from typing import Union

import d4rl
import gym
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.invdynamic import FancyMlpInvDynamic
from cleandiffuser.utils import set_seed
import json


class InvDynTrainingWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Union[D4RLMuJoCoTDDataset, D4RLAntmazeTDDataset, D4RLKitchenTDDataset],
    ):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        obs = self.dataset.obs[idx]
        act = self.dataset.act[idx]
        next_obs = self.dataset.next_obs[idx]
        return {"obs": obs, "act": act, "next_obs": next_obs}


# --- config ---
seed = 0
env_name = "antmaze-large-play-v2"
save_path = os.path.expanduser("~") + f"/.CleanDiffuser/pretrained/invdyn/{env_name}/"
save_path = Path(save_path)
devices = [1]

if "antmaze" in env_name:
    domain = "antmaze"
elif "kitchen" in env_name:
    domain = "kitchen"
else:
    domain = "mujoco"

if __name__ == "__main__":
    set_seed(seed)
    save_path.mkdir(parents=True, exist_ok=True)

    if domain == "mujoco":
        dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(gym.make(env_name)))
    elif domain == "kitchen":
        dataset = D4RLKitchenTDDataset(d4rl.qlearning_dataset(gym.make(env_name)))
    elif domain == "antmaze":
        dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(gym.make(env_name)))
    else:
        raise NotImplementedError(f"domain={domain} is not supported.")

    normalizer = dataset.normalizers["state"]
    mean = normalizer.mean
    std = normalizer.std

    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim
    dataset = InvDynTrainingWrapper(dataset)
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size])

    train_loader = DataLoader(
        train_set, batch_size=512, shuffle=True, num_workers=8, persistent_workers=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=512, shuffle=False, num_workers=8, persistent_workers=True
    )

    model = FancyMlpInvDynamic(obs_dim, act_dim)

    params = {
        "normalizers": {"state": {"mean": mean.tolist(), "std": std.tolist()}},
        "config": {
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_dim": 512,
            "tanh_out_activation": True,
            "action_scale": 1.0,
            "add_norm": True,
            "add_dropout": True,
        },
    }
    with open(save_path / "params.json", "w") as f:
        json.dump(params, f)

    earlystop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
    )
    ckpt_callback = ModelCheckpoint(dirpath=save_path)
    trainer = L.Trainer(
        devices=devices,
        callbacks=[earlystop_callback, ckpt_callback],
        default_root_dir=save_path,
    )

    trainer.fit(model, train_loader, valid_loader)
