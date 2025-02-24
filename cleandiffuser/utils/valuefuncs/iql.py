import json
import os
from copy import deepcopy
from pathlib import Path

import pytorch_lightning as L
import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: int = 256, use_layer_norm: bool = True
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Qfuncs(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, n_ensembles: int = 10):
        super().__init__()
        self.q_funcs = nn.ModuleList(
            [Mlp(obs_dim + act_dim, 1, hidden_dim) for _ in range(n_ensembles)]
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], -1)
        q = torch.stack([q_func(x) for q_func in self.q_funcs], 1)
        return q

    def min_q(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q = self.forward(obs, act)
        return torch.min(q, 1)[0]


class Vfuncs(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_ensembles: int = 1):
        super().__init__()
        self.v_funcs = nn.ModuleList([Mlp(obs_dim, 1, hidden_dim) for _ in range(n_ensembles)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        v = torch.stack([v_func(obs) for v_func in self.v_funcs], 1)
        return v

    def min_v(self, obs: torch.Tensor) -> torch.Tensor:
        v = self.forward(obs)
        return torch.min(v, 1)[0]


class Iql(L.LightningModule):
    """TODO"""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        tau: float = 0.7,
        discount: float = 0.99,
        hidden_dim: int = 512,
        q_ensembles: int = 10,
        v_ensembles: int = 1,
        ema_ratio: float = 0.99,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.lr = lr
        self.iql_tau, self.discount = tau, discount
        self.ema_ratio = ema_ratio
        self.q = Qfuncs(obs_dim, act_dim, hidden_dim, q_ensembles)
        self.v = Vfuncs(obs_dim, hidden_dim, v_ensembles)

        self.q_targ = deepcopy(self.q).requires_grad_(False).eval()

    def forward_q(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        use_ema: bool = False,
        requires_grad: bool = False,
    ):
        with torch.set_grad_enabled(requires_grad):
            if use_ema:
                q = self.q_targ.min_q(obs, act)
            else:
                q = self.q.min_q(obs, act)
        return q

    def forward_v(self, obs: torch.Tensor, requires_grad: bool = False):
        with torch.set_grad_enabled(requires_grad):
            v = self.v.min_v(obs)
        return v

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def update_target(self):
        for p, p_targ in zip(self.q.parameters(), self.q_targ.parameters()):
            p_targ.data = self.ema_ratio * p_targ.data + (1 - self.ema_ratio) * p.data

    def training_step(self, batch, batch_idx):
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        act = batch["act"]
        rew = batch["rew"]
        done = batch["tml"]

        # update V
        self.q_targ.eval()
        q = self.q_targ.min_q(obs, act)
        v = self.v.min_v(obs)
        loss_v = (torch.abs(self.iql_tau - ((q - v) < 0).to(q.dtype)) * (q - v) ** 2).mean()
        self.log("loss_v", loss_v, prog_bar=True)
        self.log("pred_v", v.mean().detach(), prog_bar=True)

        # update Q
        self.v.eval()
        with torch.no_grad():
            td_target = rew + self.discount * (1 - done) * self.v.min_v(next_obs)
            td_target = td_target.unsqueeze(1)
        pred_q = self.q(obs, act)
        loss_q = (pred_q - td_target).pow(2).mean()
        self.log("loss_q", loss_q, prog_bar=True)
        self.v.train()

        self.update_target()

        return loss_q + loss_v

    @classmethod
    def from_pretrained(self, env_name: str):
        try:
            path = os.path.expanduser("~") + f"/.CleanDiffuser/pretrained/iql/{env_name}/"
            path = Path(path)
            file_list = os.listdir(path)
            for each in file_list:
                if ".ckpt" in each:
                    print(f"Pretrained model loaded from {path / each}")
                    with open(path / "params.json", "r") as f:
                        params = json.load(f)
                    model = Iql(**params["config"])
                    model.load_state_dict(torch.load(path / each, map_location="cpu")["state_dict"])
                    return model, params
            else:
                Warning(f"No pretrained model found in {path}")
                return None, None
        except Exception as e:
            print(e)
            return None, None


if __name__ == "__main__":
    import argparse
    from typing import Union

    import d4rl
    import gym
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader

    from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
    from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
    from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
    from cleandiffuser.utils import set_seed

    class IqlTrainingWrapper(torch.utils.data.Dataset):
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
            data = self.dataset[idx]

            return {
                "obs": data["obs"]["state"],
                "next_obs": data["next_obs"]["state"],
                "act": data["act"],
                "rew": data["rew"],
                "tml": data["tml"],
            }

    # --- config ---

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--env_name", type=str, default="halfcheetah-medium-expert-v2")
    args = argparser.parse_args()

    seed = args.seed
    env_name = args.env_name
    save_path = os.path.expanduser("~") + f"/.CleanDiffuser/pretrained/iql/{env_name}/"
    save_path = Path(save_path)
    devices = [0]

    if "antmaze" in env_name:
        domain = "antmaze"
    elif "kitchen" in env_name:
        domain = "kitchen"
    else:
        domain = "mujoco"

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
    dataset = IqlTrainingWrapper(dataset)
    dataloader = DataLoader(
        dataset, batch_size=512, shuffle=True, num_workers=8, persistent_workers=True
    )

    params = {
        "normalizers": {"state": {"mean": mean.tolist(), "std": std.tolist()}},
        "config": {
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "tau": 0.9 if domain == "antmaze" else 0.7,
            "discount": 0.99,
            "hidden_dim": 512,
            "q_ensembles": 2,
            "v_ensembles": 1,
            "ema_ratio": 0.99,
            "lr": 3e-4,
        },
    }
    with open(save_path / "params.json", "w") as f:
        json.dump(params, f)

    model = Iql(**params["config"])

    ckpt_callback = ModelCheckpoint(dirpath=save_path, filename="iql-{step}")
    trainer = L.Trainer(
        devices=devices,
        callbacks=[ckpt_callback],
        default_root_dir=save_path,
        max_steps=300000 if env_name == "antmaze-large-play-v2" else 1000000,
    )

    trainer.fit(model, dataloader)
