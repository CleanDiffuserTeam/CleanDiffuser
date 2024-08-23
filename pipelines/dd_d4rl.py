from pathlib import Path

import d4rl
import gym
import hydra
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.invdynamic import FancyMlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d

RETURN_SCALE = {
    "halfcheetah-medium-expert-v2": 1200,
    "halfcheetah-medium-replay-v2": 550,
    "halfcheetah-medium-v2": 580,
    "hopper-medium-expert-v2": 400,
    "hopper-medium-replay-v2": 340,
    "hopper-medium-v2": 350,
    "walker2d-medium-expert-v2": 550,
    "walker2d-medium-replay-v2": 470,
    "walker2d-medium-v2": 480,
    "kitchen-partial-v0": 270,
    "kitchen-mixed-v0": 220,
    "antmaze-medium-play-v2": 100,
    "antmaze-medium-diverse-v2": 100,
    "antmaze-large-play-v2": 100,
    "antmaze-large-diverse-v2": 100,
}


class ObsSequence_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, env_name: str):
        self.dataset = dataset
        self.env_name = env_name
        self.scale = RETURN_SCALE[env_name]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        path_idx, start, end = self.indices[idx]
        obs = self.seq_obs[path_idx, start:end]
        val = self.seq_val[path_idx, start] / self.scale
        if "antmaze" in self.env_name:
            val += 1.
        return {
            "x0": obs,
            "condition_cfg": val, }


class InvDyn_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        path_idx, start, end = self.indices[idx]
        obs = self.seq_obs[path_idx, start:end]
        act = self.seq_act[path_idx, start:end]
        return obs[:, :-1], act[:, :-1], obs[:, 1:]


@hydra.main(config_path="../configs/dd", config_name="d4rl", version_base=None)
def pipeline(args):

    L.seed_everything(args.seed, workers=True)

    env_name = args.task.env_name
    save_path = Path(__file__).parents[1] / \
        f"results/{args.pipeline_name}/{env_name}/"

    # --- Create Dataset ---
    env = gym.make(env_name)
    raw_dataset = env.get_dataset()
    if "kitchen" in env_name:
        dataset = D4RLKitchenDataset(
            raw_dataset, horizon=args.task.horizon, discount=0.99)
    elif "antmaze" in env_name:
        dataset = D4RLAntmazeDataset(
            raw_dataset, horizon=args.task.horizon, discount=0.99)
    else:
        dataset = D4RLMuJoCoDataset(
            raw_dataset, horizon=args.task.horizon, discount=0.99)
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    # --- Create Diffusion Model ---
    nn_diffusion = DiT1d(
        in_dim=obs_dim, emb_dim=128, d_model=320, n_heads=10, depth=2,
        timestep_emb_type="untrainable_fourier")
    nn_condition = MLPCondition(
        in_dim=1, out_dim=128, hidden_dims=[128, ], act=torch.nn.SiLU(), dropout=0.25)

    # --- Create Masks & Weights ---
    fix_mask = torch.zeros((args.task.horizon, obs_dim))
    fix_mask[0] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim))
    loss_weight[1] = 10.

    # --- Diffusion Training ---
    if args.mode == "diffusion_training":

        actor = ContinuousDiffusionSDE(
            nn_diffusion, nn_condition, fix_mask, loss_weight,
            ema_rate=0.9999)

        dataloader = DataLoader(
            ObsSequence_Wrapper(dataset, env_name),
            batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)

        callback = ModelCheckpoint(
            dirpath=save_path, filename="diffusion-{step}",
            every_n_train_steps=args.save_interval)

        trainer = L.Trainer(
            accelerator='gpu', devices=[args.device_id,],
            max_steps=args.diffusion_training_steps, deterministic=True, log_every_n_steps=1000,
            default_root_dir=save_path, callbacks=[callback])

        trainer.fit(actor, dataloader)

    # --- Inverse Dynamics Training ---
    elif args.mode == "invdyn_training":

        invdyn = FancyMlpInvDynamic(
            obs_dim, act_dim, hidden_dim=args.invdyn_hidden_dim, add_dropout=True, add_norm=True)

        dataloader = DataLoader(
            InvDyn_Wrapper(dataset),
            batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)

        callback = ModelCheckpoint(
            dirpath=save_path, filename="invdyn-{step}",
            every_n_train_steps=args.save_interval)

        trainer = L.Trainer(
            accelerator='gpu', devices=[args.device_id,],
            max_steps=args.invdyn_training_steps, deterministic=True, log_every_n_steps=1000,
            default_root_dir=save_path, callbacks=[callback])

        trainer.fit(invdyn, dataloader)

    # --- Inference ---
    elif args.mode == "inference":

        num_envs = args.num_envs
        num_episodes = args.num_episodes
        num_candidates = args.num_candidates

        if args.invdyn_from_pretrain:
            invdyn = FancyMlpInvDynamic.from_pretrained(
                env_name, hidden_dim=args.invdyn_hidden_dim)
        else:
            invdyn = FancyMlpInvDynamic.load_from_checkpoint(
                checkpoint_path=save_path / "invdyn-step=300000.ckpt",
                obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.invdyn_hidden_dim)
        invdyn.to(f"cuda:{args.device_id}").eval()

        actor = ContinuousDiffusionSDE.load_from_checkpoint(
            checkpoint_path=save_path / "diffusion-step=300000.ckpt",
            nn_diffusion=nn_diffusion, nn_condition=nn_condition,
            fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=0.9999)
        actor.to(f"cuda:{args.device_id}").eval()

        env_eval = gym.vector.make(env_name, num_envs=num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((num_envs, args.task.horizon, obs_dim))
        for i in range(num_episodes):

            obs, ep_reward, all_done, t = env_eval.reset(), 0., False, 0

            while not np.all(all_done) and t < 1000:

                pass


if __name__ == "__main__":
    pipeline()
