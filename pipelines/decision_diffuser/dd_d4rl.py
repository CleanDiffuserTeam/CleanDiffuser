import argparse
from pathlib import Path

import d4rl  # noqa: F401
import gym
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from termcolor import cprint
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.invdynamic import FancyMlpInvDynamic
from cleandiffuser.nn_condition import FourierCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import set_seed

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
            val += 1.0
        return {
            "x0": obs,
            "condition_cfg": val,
        }


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
        return obs[:-1], act[:-1], obs[1:]


# --- config ---
argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=0, help="random seed")
argparser.add_argument(
    "--env_name", type=str, default="halfcheetah-medium-expert-v2", help="env name"
)
argparser.add_argument("--horizon", type=int, default=32, help="planning horizon")
argparser.add_argument(
    "--mode",
    type=str,
    default="diffusion_training",
    help="Mode: diffusion_training, invdyn_training, inference",
)
argparser.add_argument(
    "--devices",
    type=int,
    nargs="+",
    default=[0, 1],
    help="Devices used for training, e.g. 0 1. only the first device is used for inference.",
)
argparser.add_argument("--training_steps", type=int, default=500_000, help="training steps")
argparser.add_argument("--save_every_n_steps", type=int, default=100_000, help="save interval")
argparser.add_argument(
    "--ckpt_file",
    type=str,
    default="diffusion-step=500000.ckpt",
    help="ckpt file used for inference",
)
argparser.add_argument("--sampling_steps", type=int, default=10, help="sampling steps")
argparser.add_argument(
    "--num_envs", type=int, default=50, help="number of parallel envs for evaluation"
)
argparser.add_argument("--num_episodes", type=int, default=3, help="number of seeds for evaluation")
argparser.add_argument(
    "--target_return",
    type=float,
    default=None,
    help="target return. set this value to overwrite the default value.",
)
argparser.add_argument(
    "--w_cfg",
    type=float,
    default=None,
    help="w_cfg. set this value to overwrite the default value.",
)
args = argparser.parse_args()

seed = args.seed
env_name = args.env_name
horizon = args.horizon
mode = args.mode
devices = args.devices
training_steps = args.training_steps
save_every_n_steps = args.save_every_n_steps
ckpt_file = args.ckpt_file
sampling_steps = args.sampling_steps
num_envs = args.num_envs
num_episodes = args.num_episodes

if env_name == "halfcheetah-medium-expert-v2":
    target_return = 1.0
    w_cfg = 1.2
elif env_name == "halfcheetah-medium-v2":
    target_return = 1.0
    w_cfg = 3.0
elif env_name == "halfcheetah-medium-replay-v2":
    target_return = 0.95
    w_cfg = 3.0
elif env_name == "hopper-medium-expert-v2":
    target_return = 0.95
    w_cfg = 1.4
elif env_name == "hopper-medium-v2":
    target_return = 0.9
    w_cfg = 7.0
elif env_name == "hopper-medium-replay-v2":
    target_return = 0.9
    w_cfg = 3.0
elif env_name == "walker2d-medium-expert-v2":
    target_return = 1.0
    w_cfg = 5.0
elif env_name == "walker2d-medium-v2":
    target_return = 0.8
    w_cfg = 6.0
elif env_name == "walker2d-medium-replay-v2":
    target_return = 0.9
    w_cfg = 2.0
elif env_name == "kitchen-mixed-v0":
    target_return = 0.85
    w_cfg = 1.0
    sampling_steps = 5 if sampling_steps == 10 else sampling_steps  # 5 as default
elif env_name == "kitchen-partial-v0":
    target_return = 0.85
    w_cfg = 1.0
    sampling_steps = 5 if sampling_steps == 10 else sampling_steps  # 5 as default
elif env_name == "antmaze-medium-play-v2":
    target_return = 0.3
    w_cfg = 2.5
elif env_name == "antmaze-medium-diverse-v2":
    target_return = 0.3
    w_cfg = 7.0
elif env_name == "antmaze-large-play-v2":
    target_return = 0.3
    w_cfg = 12.0
elif env_name == "antmaze-large-diverse-v2":  # no hparam works :(
    target_return = 0.3
    w_cfg = 1.0
else:
    raise NotImplementedError(f"env_name={env_name} is not implemented.")

target_return = args.target_return or target_return
w_cfg = args.w_cfg or w_cfg


if __name__ == "__main__":
    set_seed(seed)

    save_path = Path(__file__).parents[2] / f"results/decision_diffuser/{env_name}/"

    # --- Dataset ---
    env = gym.make(env_name)
    if "kitchen" in env_name:
        dataset = D4RLKitchenDataset(env.get_dataset(), horizon=horizon, discount=0.99)
    elif "antmaze" in env_name:
        dataset = D4RLAntmazeDataset(env.get_dataset(), horizon=horizon, discount=0.99)
    else:
        dataset = D4RLMuJoCoDataset(env.get_dataset(), horizon=horizon, discount=0.99)
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    # --- Create Diffusion Model ---
    nn_diffusion = DiT1d(
        x_dim=obs_dim,
        x_seq_len=horizon,
        emb_dim=128,
        d_model=256,
        n_heads=8,
        depth=4,
        timestep_emb_type="untrainable_fourier",
    )
    nn_condition = FourierCondition(
        in_dim=128, out_dim=128, hidden_dims=128, dropout=0.25, scale=0.2
    )

    # --- Create Masks & Weights ---
    fix_mask = torch.zeros((horizon, obs_dim))
    fix_mask[0] = 1.0
    loss_weight = torch.ones((horizon, obs_dim))
    loss_weight[1] = 10.0

    actor = ContinuousDiffusionSDE(
        nn_diffusion, nn_condition, fix_mask, loss_weight, ema_rate=0.999
    )

    # --- Diffusion Training ---
    if mode == "diffusion_training":
        dataloader = DataLoader(
            ObsSequence_Wrapper(dataset, env_name),
            batch_size=512,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

        callback = ModelCheckpoint(
            dirpath=save_path,
            filename="diffusion-{step}",
            every_n_train_steps=save_every_n_steps,
            save_top_k=-1,
        )

        trainer = L.Trainer(
            devices=devices,
            max_steps=training_steps,
            default_root_dir=save_path,
            callbacks=[callback],
        )

        trainer.fit(actor, dataloader)

    # --- Inverse Dynamics Training ---
    elif mode == "invdyn_training":
        cprint("Please run `cleandiffuser/invdynamic/pretrain_on_d4rl.py`!", "green")

    # --- Inference ---
    elif mode == "inference":
        device = f"cuda:{devices[0]}"

        invdyn, invdyn_params = FancyMlpInvDynamic.from_pretrained(env_name)
        invdyn.to(device).eval()

        actor.load_state_dict(torch.load(save_path / ckpt_file, map_location=device)["state_dict"])
        actor = actor.to(device).eval()

        env_eval = gym.vector.make(env_name, num_envs=num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((num_envs, horizon, obs_dim))
        condition = torch.ones((num_envs, 1), device=device) * target_return
        for i in range(num_episodes):
            obs, ep_reward, all_done, t = env_eval.reset(), 0.0, False, 0

            while not np.all(all_done) and t < 1000:
                obs = torch.tensor(normalizer.normalize(obs), dtype=torch.float32)
                prior[:, 0] = obs

                traj, log = actor.sample(
                    prior,
                    solver="ddpm",
                    sample_steps=sampling_steps,
                    condition_cfg=condition,
                    w_cfg=w_cfg,
                    sampling_schedule="quad",
                )

                with torch.no_grad():
                    act = invdyn(traj[:, 0], traj[:, 1]).cpu().numpy()

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

    else:
        raise NotImplementedError(f"mode={mode} is not implemented.")
