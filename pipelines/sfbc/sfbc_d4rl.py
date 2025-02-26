import argparse
from pathlib import Path
from typing import Union

import d4rl
import einops
import gym
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from termcolor import cprint
from torch.utils.data import DataLoader
from tqdm import tqdm

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeDataset, D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenDataset, D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset, D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import GaussianNormalizer, dict_apply, loop_dataloader, set_seed


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        return self.model(torch.cat([obs, act], dim=-1))


class InsamplePlanningD4RLDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Union[D4RLMuJoCoDataset, D4RLAntmazeDataset, D4RLKitchenDataset],
    ):
        super().__init__()
        self.dataset = dataset
        self.val_normalizer = GaussianNormalizer(self.seq_val)
        self.normed_seq_val = self.val_normalizer.normalize(self.seq_val)

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]
        data = {
            "obs": {"state": self.seq_obs[path_idx, start:end]},
            "act": self.seq_act[path_idx, start:end],
            "rew": self.seq_rew[path_idx, start:end],
            "val": self.normed_seq_val[path_idx, start:end],
        }
        return dict_apply(data, torch.tensor)


class BC_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        return {
            "x0": self.act[idx],
            "condition_cfg": self.obs[idx],
        }


# -- config --

argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--env_name", type=str, default="halfcheetah-medium-expert-v2")
argparser.add_argument("--mode", type=str, default="training")
argparser.add_argument("--save_every_n_steps", type=int, default=200000)
argparser.add_argument("--training_steps", type=int, default=1000000)
argparser.add_argument("--ckpt_file", type=str, default="diffusion_bc-step=1000000.ckpt")
argparser.add_argument("--num_envs", type=int, default=50)
argparser.add_argument("--num_episodes", type=int, default=3)
argparser.add_argument("--num_candidates", type=int, default=50)
argparser.add_argument("--M", type=int, default=16)
argparser.add_argument("--alpha", type=int, default=20)
argparser.add_argument("--sampling_steps", type=int, default=5)
argparser.add_argument("--devices", type=int, nargs="+", default=[0])
argparser.add_argument("--q_training_iters", type=int, default=-1)
argparser.add_argument("--critic_training_steps", type=int, default=10000)
argparser.add_argument("--critic_lr", type=float, default=1e-3)
args = argparser.parse_args()

seed = args.seed
env_name = args.env_name
M = args.M
alpha = args.alpha
devices = args.devices
mode = args.mode
sampling_steps = args.sampling_steps
save_every_n_steps = args.save_every_n_steps
training_steps = args.training_steps
ckpt_file = args.ckpt_file
q_training_iters = args.q_training_iters
critic_training_steps = args.critic_training_steps
critic_lr = args.critic_lr
num_envs = args.num_envs
num_episodes = args.num_episodes
num_candidates = args.num_candidates

# default
if q_training_iters == -1:
    if "antmaze" in env_name or "kitchen" in env_name:
        q_training_iters = 5
    else:
        q_training_iters = 2

if __name__ == "__main__":
    set_seed(seed)
    save_path = Path(__file__).parents[2] / f"results/sfbc/{env_name}/"

    # --- Create Dataset ---
    env = gym.make(env_name)
    if "kitchen" in env_name:
        dataset = D4RLKitchenTDDataset(d4rl.qlearning_dataset(env))
    elif "antmaze" in env_name:
        dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(env))
    else:
        dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env))
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    # --- Create Diffusion Model ---
    nn_diffusion = IDQLMlp(x_dim=act_dim, timestep_emb_type="untrainable_fourier")
    nn_condition = MLPCondition(in_dim=obs_dim, out_dim=64, hidden_dims=64, dropout=0.0)

    actor = ContinuousDiffusionSDE(
        nn_diffusion,
        nn_condition,
        ema_rate=0.999,
        x_max=+1.0 * torch.ones((act_dim,)),
        x_min=-1.0 * torch.ones((act_dim,)),
    )

    # --- BC Training ---
    if mode == "bc_training":
        dataloader = DataLoader(
            BC_Wrapper(dataset),
            batch_size=512,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

        callback = ModelCheckpoint(
            dirpath=save_path,
            filename="diffusion_bc-{step}",
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

    elif mode == "critic_training":
        device = f"cuda:{devices[0]}"

        actor.load_state_dict(torch.load(save_path / ckpt_file, map_location=device)["state_dict"])
        actor.to(device).eval()

        critic = None  # wait for initialization

        if "kitchen" in env_name:
            dataset = D4RLKitchenDataset(env.get_dataset(), horizon=32)
        elif "antmaze" in env_name:
            dataset = D4RLAntmazeDataset(env.get_dataset(), horizon=32)
        else:
            dataset = D4RLMuJoCoDataset(env.get_dataset(), horizon=32)

        dataset = InsamplePlanningD4RLDataset(dataset)

        dataloader = DataLoader(
            dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True
        )

        max_traj_len = dataset.seq_obs.shape[1]

        for k in range(q_training_iters):
            cprint(f"Critic training iteration: {k + 1} ...", "green")
            if k > 0:
                cprint("Relabeling values ...", "cyan")
                critic.eval()
                # M Monte Carlo evaluation
                normed_eval_seq_val = np.empty_like(dataset.normed_seq_val)
                for i in tqdm(range(dataset.normed_seq_val.shape[0])):
                    obs = (
                        torch.tensor(dataset.seq_obs[i], device=device).unsqueeze(1).repeat(1, M, 1)
                    )
                    prior = torch.zeros((max_traj_len * M, act_dim))
                    act = actor.sample(
                        prior,
                        solver="ddpm",
                        sample_steps=sampling_steps,
                        condition_cfg=obs.view(-1, obs_dim),
                        w_cfg=1.0,
                        sample_step_schedule="quad",
                    )[0].view(-1, M, act_dim)

                    with torch.no_grad():
                        pred_val = critic(obs, act)

                    weight = torch.nn.functional.softmax(alpha * pred_val, dim=1)
                    normed_eval_seq_val[i] = (weight * pred_val).sum(1).cpu().numpy()

                # Implicit in-sample planning
                eval_seq_val = dataset.val_normalizer.unnormalize(normed_eval_seq_val)

                target_seq_val = np.empty_like(eval_seq_val)
                target_seq_val[:, :-1] = dataset.seq_rew[:, :-1] + 0.99 * np.maximum(
                    dataset.seq_val[:, 1:], eval_seq_val[:, 1:]
                )
                target_seq_val[:, -1] = eval_seq_val[:, -1]
                if dataset.tml_and_not_timeout.shape[0] != 0:
                    target_seq_val[np.where(dataset.tml_and_not_timeout)] = (dataset.seq_rew)[
                        np.where(dataset.tml_and_not_timeout)
                    ]
                dataset.seq_val = target_seq_val
                dataset.val_normalizer.__init__(target_seq_val)
                dataset.normed_seq_val = dataset.val_normalizer.normalize(target_seq_val)
                dataloader = DataLoader(
                    dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True
                )

            # Critic training, reset critic for each iteration
            n_gradient_step = 0
            critic = Critic(obs_dim, act_dim).to(device)
            optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)

            cprint("Critic training ...", "cyan")
            log = dict.fromkeys(["critic_loss"], 0.0)
            for batch in loop_dataloader(dataloader):
                obs, act, val = (
                    batch["obs"]["state"].to(device),
                    batch["act"].to(device),
                    batch["val"].to(device),
                )
                critic_loss = (critic(obs, act) - val).pow(2).mean()
                optim.zero_grad()
                critic_loss.backward()
                optim.step()
                log["critic_loss"] += critic_loss.item()
                if (n_gradient_step + 1) % 500 == 0:
                    log = {k: v / 500 for k, v in log.items()}
                    log["gradient_step"] = n_gradient_step + 1
                    print(log)
                    log = dict.fromkeys(["critic_loss"], 0.0)
                n_gradient_step += 1
                if n_gradient_step > critic_training_steps:
                    break

        cprint("Done!", "cyan")
        torch.save(critic.state_dict(), save_path / "critic.pt")

    elif mode == "inference":
        device = f"cuda:{devices[0]}"

        critic = Critic(obs_dim, act_dim).to(device).eval()
        critic.load_state_dict(torch.load(save_path / "critic.pt", map_location=device))

        actor.load_state_dict(torch.load(save_path / ckpt_file, map_location=device)["state_dict"])
        actor = actor.to(device).eval()

        env_eval = gym.vector.make(env_name, num_envs=num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((num_envs * num_candidates, act_dim))

        for i in range(num_episodes):
            obs, ep_reward, all_done, t = env_eval.reset(), 0.0, False, 0

            while not np.all(all_done) and t < 1000:
                obs = torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32)
                obs = einops.repeat(obs, "b d -> (b k) d", k=num_candidates)

                act, _ = actor.sample(
                    prior,
                    solver="ddpm",
                    sample_steps=sampling_steps,
                    condition_cfg=obs,
                    w_cfg=1.0,
                    sample_step_schedule="quad",
                )

                with torch.no_grad():
                    value = critic(obs, act)
                    value = einops.rearrange(value, "(b k) 1 -> b k 1", k=num_candidates)
                    act = einops.rearrange(act, "(b k) d -> b k d", k=num_candidates)
                    idx = value.argsort(dim=1, descending=True).squeeze(-1)
                    act = act[torch.arange(act.size(0)), idx[:, 0]]
                    act = act.cpu().numpy()

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
