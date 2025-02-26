import argparse
from pathlib import Path
from typing import Dict, Union

import d4rl
import gym
import h5py
import numpy as np
import pytorch_lightning as L
import torch
import torch.utils
import torch.utils.data
from pytorch_lightning.callbacks import ModelCheckpoint
from termcolor import cprint
from torch.utils.data import DataLoader
from tqdm import tqdm

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import set_seed
from cleandiffuser.utils.offlinerl.td3bc import TD3BC


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


class DatasetMergeWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Union[D4RLMuJoCoTDDataset, D4RLAntmazeTDDataset, D4RLKitchenTDDataset],
        upsampled_dataset: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.dataset = dataset
        self.obs = torch.cat([dataset.obs, upsampled_dataset["obs"]], 0)
        self.next_obs = torch.cat([dataset.next_obs, upsampled_dataset["next_obs"]], 0)
        self.act = torch.cat([dataset.act, upsampled_dataset["act"]], 0)
        self.rew = torch.cat([dataset.rew, upsampled_dataset["rew"]], 0)
        self.tml = torch.cat([dataset.tml, upsampled_dataset["tml"]], 0)
        self.size = self.obs.shape[0]

    def __len__(self):
        return self.size

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        return {
            "obs": {"state": self.obs[idx]},
            "next_obs": {"state": self.next_obs[idx]},
            "act": self.act[idx],
            "rew": self.rew[idx],
            "tml": self.tml[idx],
        }


# --- config ---
argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--env_name", type=str, default="hopper-medium-v2")
argparser.add_argument("--mode", type=str, default="synther_training")
argparser.add_argument("--save_every_n_steps", type=int, default=100_000)
argparser.add_argument("--training_steps", type=int, default=500_000)
argparser.add_argument("--td3bc_save_every_n_steps", type=int, default=200_000)
argparser.add_argument("--td3bc_training_steps", type=int, default=1000_000)
argparser.add_argument("--devices", type=int, nargs="+", default=[0])
argparser.add_argument("--ckpt_file", type=str, default="synther-step=500000.ckpt")
argparser.add_argument("--td3bc_ckpt_file", type=str, default="td3bc-step=1000000.ckpt")
argparser.add_argument("--upsampling_size", type=int, default=2000000)
argparser.add_argument("--num_envs", type=int, default=50)
argparser.add_argument("--num_episodes", type=int, default=3)
args = argparser.parse_args()

seed = args.seed
env_name = args.env_name
mode = args.mode
save_every_n_steps = args.save_every_n_steps
training_steps = args.training_steps
td3bc_save_every_n_steps = args.td3bc_save_every_n_steps
td3bc_training_steps = args.td3bc_training_steps
devices = args.devices
ckpt_file = args.ckpt_file
td3bc_ckpt_file = args.td3bc_ckpt_file
upsampling_size = args.upsampling_size
num_envs = args.num_envs
num_episodes = args.num_episodes


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
    elif mode == "dataset_upsampling":
        device = f"cuda:{devices[0]}"

        synther.load_state_dict(
            torch.load(save_path / ckpt_file, map_location=device)["state_dict"]
        )

        ori_size = dataset.size
        syn_size = upsampling_size - ori_size
        max_batch_size = 20000

        syn_obs = np.empty((syn_size, obs_dim))
        syn_next_obs = np.empty((syn_size, obs_dim))
        syn_rew = np.empty((syn_size, 1))
        syn_act = np.empty((syn_size, act_dim))
        syn_tml = np.empty((syn_size, 1))

        cprint(f"Total dataset size: {ori_size + syn_size}", color="cyan")
        cprint(f"Original dataset size: {ori_size}", color="cyan")
        cprint(f"Synthetic dataset size: {syn_size}", color="cyan")
        cprint(f"Batch size: {max_batch_size}", color="cyan")
        cprint("Begin upsampling...", color="cyan")

        prior, ptr = torch.zeros((max_batch_size, x_dim)), 0
        for i in tqdm(range(0, syn_size, max_batch_size)):
            batch_size = min(syn_size - i, max_batch_size)

            transition, _ = synther.sample(
                prior[:batch_size], solver="ddpm", sample_steps=20, sampling_schedule="quad"
            )
            transition = transition.cpu().numpy()

            syn_obs[ptr : ptr + batch_size] = transition[:, :obs_dim]
            syn_next_obs[ptr : ptr + batch_size] = transition[:, obs_dim : 2 * obs_dim]
            syn_rew[ptr : ptr + batch_size] = transition[:, 2 * obs_dim : 2 * obs_dim + 1]
            syn_act[ptr : ptr + batch_size] = transition[
                :, 2 * obs_dim + 1 : 2 * obs_dim + 1 + act_dim
            ]
            syn_tml[ptr : ptr + batch_size] = transition[:, -1:]
            ptr += batch_size

        assert ptr == syn_size

        upsampled_dataset = {
            "obs": syn_obs,
            "next_obs": syn_next_obs,
            "rew": syn_rew,
            "act": syn_act,
            "tml": syn_tml,
        }

        with h5py.File(save_path / "upsampled_dataset.hdf5", "w") as f:
            for k, v in upsampled_dataset.items():
                f.create_dataset(k, data=v)

        print(f"Upsampling done. Saved to {save_path / 'upsampled_dataset.hdf5'}")

    elif mode == "td3bc_training":
        upsampled_dataset = dict()
        with h5py.File(save_path / "upsampled_dataset.hdf5", "r") as f:
            upsampled_dataset["act"] = torch.tensor(
                np.clip(f["act"][:], -1.0, 1.0).astype(np.float32)
            )
            upsampled_dataset["obs"] = torch.tensor(f["obs"][:], dtype=torch.float32)
            upsampled_dataset["next_obs"] = torch.tensor(f["next_obs"][:], dtype=torch.float32)
            upsampled_dataset["rew"] = torch.tensor(f["rew"][:], dtype=torch.float32)
            upsampled_dataset["tml"] = torch.tensor(f["tml"][:] > 0.5, dtype=torch.float32)

        td3bc = TD3BC(obs_dim, act_dim)

        dataloader = DataLoader(
            DatasetMergeWrapper(dataset, upsampled_dataset),
            batch_size=256,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

        callback = ModelCheckpoint(
            dirpath=save_path,
            filename="td3bc-{step}",
            every_n_train_steps=td3bc_save_every_n_steps,
            save_top_k=-1,
        )

        trainer = L.Trainer(
            devices=devices,
            max_steps=td3bc_training_steps,
            default_root_dir=save_path,
            callbacks=[callback],
        )

        trainer.fit(td3bc, dataloader)

    elif mode == "inference":
        device = f"cuda:{devices[0]}"
        actor = TD3BC(obs_dim, act_dim)

        actor.load_state_dict(
            torch.load(save_path / td3bc_ckpt_file, map_location=device)["state_dict"]
        )
        actor = actor.to(device).eval()

        env_eval = gym.vector.make(env_name, num_envs=num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        for i in range(num_episodes):
            obs, ep_reward, all_done, t = env_eval.reset(), 0.0, False, 0

            while not np.all(all_done) and t < 1000:
                act = actor.act(normalizer.normalize(obs.astype(np.float32)))
                obs, rew, done, _ = env_eval.step(act)

                t += 1
                done = np.logical_and(done, t < 1000)
                all_done = np.logical_or(all_done, done)
                if "kitchen" in env_name:
                    ep_reward = np.clip(ep_reward + rew, 0.0, 4.0)
                    print(f"[t={t}] finished tasks: {np.around(ep_reward)}")
                elif "antmaze" in env_name:
                    ep_reward = np.clip(ep_reward + rew, 0.0, 1.0)
                    print(f"[t={t}] xy: {np.around(obs[:, :2], 2)}")
                    print(f"[t={t}] reached goal: {np.around(ep_reward)}")
                else:
                    ep_reward += rew * (1 - all_done)
                    print(f"[t={t}] rew: {np.around((rew * (1 - all_done)), 2)}")

                if np.all(all_done):
                    break

            episode_rewards.append(ep_reward)

        episode_rewards = [
            list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards
        ]
        episode_rewards = np.array(episode_rewards).mean(-1) * 100.0
        print(f"Score: {episode_rewards.mean():.3f}±{episode_rewards.std():.3f}")

        env_eval.close()
