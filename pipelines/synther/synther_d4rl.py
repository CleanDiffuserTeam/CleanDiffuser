"""
WARNING: This pipeline has not been fully tested. The results may not be accurate.
You may tune the hyperparameters in the config file before using it.
"""

from pathlib import Path
from typing import Dict, Union

import d4rl
import gym
import h5py
import hydra
import numpy as np
import pytorch_lightning as L
import torch
import torch.utils
import torch.utils.data
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import set_seed


class Transition_Wrapper(torch.utils.data.Dataset):
    def __init__(
        self, dataset: Union[D4RLMuJoCoTDDataset, D4RLAntmazeTDDataset, D4RLKitchenTDDataset]
    ):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        obs = self.obs[idx]
        next_obs = self.next_obs[idx]
        act = self.act[idx]
        rew = self.rew[idx]
        tml = self.tml[idx]
        return {"x0": torch.cat([obs, next_obs, rew, act, tml], -1)}


class Upsampling_Wrapper(torch.utils.data.Dataset):
    def __init__(
        self, dataset: torch.utils.data.Dataset, upsampling_dataset: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.dataset = dataset
        self.obs = torch.cat([dataset.obs, upsampling_dataset["obs"]], 0)
        self.next_obs = torch.cat([dataset.next_obs, upsampling_dataset["next_obs"]], 0)
        self.act = torch.cat([dataset.act, upsampling_dataset["act"]], 0)
        self.rew = torch.cat([dataset.rew, upsampling_dataset["rew"]], 0)
        self.tml = torch.cat([dataset.tml, upsampling_dataset["tml"]], 0)
        self.size = self.obs.shape[0]

    def __len__(self):
        return self.size

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.tml[idx]


# --- config ---
seed = 0
env_name = "hopper-medium-v2"
mode = "synther_training"
save_every_n_steps = 100_000
training_steps = 500_000
devices = [0]

if __name__ == "__main__":
    set_seed(seed)

    save_path = Path(__file__).parents[2] / f"results/synther/{env_name}/"

    # --- Create Dataset ---
    env = gym.make(env_name)
    if "kitchen" in env_name:
        dataset = D4RLKitchenTDDataset(d4rl.qlearning_dataset(env))
    elif "antmaze" in env_name:
        dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(env))
    else:
        dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), normalize_reward=True)
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    """
    SynthER generates transitions, which are aranged as
    (obs, next_obs, rew, act, tml).
    So the dimension is `(obs_dim + obs_dim + 1 + act_dim + 1)`.
    """
    x_dim = 2 * obs_dim + act_dim + 2

    # --- Create Diffusion Model ---
    nn_diffusion = IDQLMlp(
        x_dim=x_dim,
        emb_dim=512,
        hidden_dim=512,
        n_blocks=3,
        timestep_emb_type="untrainable_fourier",
        timestep_emb_params={"scale": 0.2},
    )

    synther = ContinuousDiffusionSDE(nn_diffusion)

    # --- SynthER Training ---
    if mode == "synther_training":
        dataloader = DataLoader(
            Transition_Wrapper(dataset),
            batch_size=512,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

        callback = ModelCheckpoint(
            dirpath=save_path,
            filename="synther-{step}",
            every_n_train_steps=save_every_n_steps,
            save_top_k=-1,
        )

        trainer = L.Trainer(
            devices=devices,
            max_steps=training_steps,
            default_root_dir=save_path,
            callbacks=[callback],
        )

        trainer.fit(synther, dataloader)

    # --- Dataset Upsampling ---
    elif args.mode == "dataset_upsampling":
        synther = ContinuousDiffusionSDE(nn_diffusion, ema_rate=0.999, x_max=x_max, x_min=x_min)
        synther.load_state_dict(
            torch.load(
                save_path / f"synther-step={args.diffusion_training_steps}.ckpt",
                map_location=device,
            )["state_dict"]
        )
        synther.eval().to(device)

        ori_size = dataset.size
        syn_size = args.upsampling_size - ori_size
        max_batch_size = 5000

        syn_obs = torch.empty((syn_size, obs_dim))
        syn_next_obs = torch.empty((syn_size, obs_dim))
        syn_rew = torch.empty((syn_size, 1))
        syn_act = torch.empty((syn_size, act_dim))
        syn_tml = torch.empty((syn_size, 1))

        print(f"Total dataset size: {ori_size + syn_size}")
        print(f"Original dataset size: {ori_size}")
        print(f"Synthetic dataset size: {syn_size}")
        print(f"Batch size: {max_batch_size}")
        print("Begin upsampling...")

        prior, ptr = torch.zeros((max_batch_size, x_dim)), 0
        for i in tqdm(range(0, syn_size, max_batch_size)):
            batch_size = min(syn_size - i, max_batch_size)

            transition, _ = synther.sample(
                prior[:batch_size],
                solver=args.solver,
                n_samples=batch_size,
                sample_steps=20,
            )
            transition.cpu()

            syn_obs[ptr : ptr + batch_size] = transition[:, :obs_dim]
            syn_next_obs[ptr : ptr + batch_size] = transition[:, obs_dim : 2 * obs_dim]
            syn_rew[ptr : ptr + batch_size] = transition[:, 2 * obs_dim : 2 * obs_dim + 1]
            syn_act[ptr : ptr + batch_size] = transition[
                :, 2 * obs_dim + 1 : 2 * obs_dim + 1 + act_dim
            ]
            syn_tml[ptr : ptr + batch_size] = transition[:, -1:]
            ptr += batch_size

        assert ptr == syn_size

        upsampling_dataset = {
            "obs": syn_obs,
            "next_obs": syn_next_obs,
            "rew": syn_rew,
            "act": syn_act,
            "tml": syn_tml,
        }

        with h5py.File(save_path / "upsampling_dataset.hdf5", "w") as f:
            for k, v in upsampling_dataset.items():
                f.create_dataset(k, data=v.numpy())

        print(f"Upsampling done. Saved to {save_path / 'upsampling_dataset.hdf5'}")

    elif args.mode == "td3bc_training":
        with h5py.File(save_path / "upsampling_dataset.hdf5", "r") as f:
            upsampling_dataset = {k: torch.tensor(v[:]) for k, v in f.items()}

        td3bc = TD3BC(obs_dim, act_dim)

        dataloader = DataLoader(
            Upsampling_Wrapper(dataset, upsampling_dataset),
            batch_size=512,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

        callback = ModelCheckpoint(
            dirpath=save_path, filename="td3bc-{step}", every_n_train_steps=args.save_interval
        )

        trainer = L.Trainer(
            accelerator="gpu",
            devices=[0, 1, 2, 3],
            max_steps=args.td3bc_training_steps,
            deterministic=True,
            log_every_n_steps=200,
            default_root_dir=save_path,
            callbacks=[callback],
            strategy="ddp_find_unused_parameters_true",
        )

        trainer.fit(td3bc, dataloader)

    elif args.mode == "inference":
        td3bc = TD3BC(obs_dim, act_dim)
        td3bc.load_state_dict(
            torch.load(save_path / f"td3bc-step={args.td3bc_ckpt}.ckpt", map_location=device)[
                "state_dict"
            ]
        )
        td3bc.eval().to(device)

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        for i in range(args.num_episodes):
            obs, ep_reward, cum_done, t = env_eval.reset(), 0.0, 0.0, 0

            while not np.all(cum_done) and t < 1000 + 1:
                # normalize obs
                obs = normalizer.normalize(obs)

                # sample actions
                act = td3bc.act(obs)

                # step
                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                print(f"[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}")

                if np.all(cum_done):
                    break

            episode_rewards.append(ep_reward)

        episode_rewards = [
            list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards
        ]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))
