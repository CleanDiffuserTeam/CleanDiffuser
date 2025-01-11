"""
WARNING: This pipeline has not been fully tested. The results may not be accurate.
You may tune the hyperparameters in the config file before using it.
"""

import os
from copy import deepcopy
from pathlib import Path

import d4rl
import einops
import gym
import hydra
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import FreezeModules, loop_dataloader


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim: int = 256):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )
        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def both(self, obs, act):
        q1, q2 = self.Q1(torch.cat([obs, act], -1)), self.Q2(torch.cat([obs, act], -1))
        return q1, q2

    def forward(self, obs, act):
        return torch.min(*self.both(obs, act))


@hydra.main(config_path="../configs/dql", config_name="d4rl", version_base=None)
def pipeline(args):
    L.seed_everything(args.seed, workers=True)

    env_name = args.task.env_name
    save_path = Path(__file__).parents[1] / f"results/{args.pipeline_name}/{env_name}/"

    device = f"cuda:{args.device_id}"

    # --- Create Dataset ---
    env = gym.make(env_name)
    raw_dataset = d4rl.qlearning_dataset(env)
    if "kitchen" in env_name:
        dataset = D4RLKitchenTDDataset(raw_dataset)
    elif "antmaze" in env_name:
        dataset = D4RLAntmazeTDDataset(raw_dataset, reward_tune="iql")
    else:
        dataset = D4RLMuJoCoTDDataset(raw_dataset, normalize_reward=True)
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    # --- Create Diffusion Model ---
    nn_diffusion = IDQLMlp(
        x_dim=act_dim, emb_dim=64, hidden_dim=256, n_blocks=3, dropout=0.0, timestep_emb_type="positional"
    )
    nn_condition = MLPCondition(
        in_dim=obs_dim,
        out_dim=64,
        hidden_dims=[64],
        act=torch.nn.SiLU(),
        dropout=0.0,
    )

    critic = TwinQ(obs_dim, act_dim, hidden_dim=256).to(device)
    critic_target = deepcopy(critic).requires_grad_(False).eval().to(device)

    # --- Training ---
    if args.mode == "training":
        """
        Unlike other pipelines, DQL training contains two coupled training steps:
        (1) policy Q function update, and (2) diffusion BC with Q maximization.
        So it is not straightforward to use the standard pytorch-lightning trainer.
        We can instead manually train the diffusion model and the policy Q function,
        which is much more readable and convenient.
        """

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        dataloader = DataLoader(
            dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True
        )

        critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)

        """
        While a high `ema_rate` performs better when using the standard
        diffusion loss, it causes instability when adding an additional
        Q maximization loss. So we use a smaller `ema_rate` here.
        """
        actor = DiscreteDiffusionSDE(
            nn_diffusion,
            nn_condition,
            ema_rate=0.995,
            diffusion_steps=args.sampling_steps,
            x_max=+1.0 * torch.ones((act_dim,)),
            x_min=-1.0 * torch.ones((act_dim,)),
        ).to(device)
        actor.configure_manual_optimizers()

        step = 0
        log = dict.fromkeys(["bc_loss", "policy_q_loss", "critic_td_loss", "target_q"], 0.0)

        prior = torch.zeros((256, act_dim), device=device)

        for batch in loop_dataloader(dataloader):
            obs, next_obs = batch["obs"]["state"].to(device), batch["next_obs"]["state"].to(device)
            act = batch["act"].to(device)
            rew = batch["rew"].to(device)
            tml = batch["tml"].to(device)

            # --- Critic Update ---
            actor.eval()
            critic.train()

            q1, q2 = critic.both(obs, act)

            """ Use max-Q backup for AntMaze, otherwise use policy-Q backup. """
            if "antmaze" in env_name:
                repeat_next_obs = einops.repeat(next_obs, "b d -> (b n) d", n=10)
                next_act, _ = actor.sample(
                    prior.repeat(10, 1),
                    solver=args.solver,
                    n_samples=2560,
                    sample_steps=args.sampling_steps,
                    condition_cfg=repeat_next_obs,
                    w_cfg=1.0,
                    requires_grad=False,
                )

                with torch.no_grad():
                    target_q1, target_q2 = critic_target.both(repeat_next_obs, next_act)
                    target_q1 = einops.rearrange(target_q1, "(b n) 1 -> b n 1", n=10).max(1)[0]
                    target_q2 = einops.rearrange(target_q2, "(b n) 1 -> b n 1", n=10).max(1)[0]
                    target_q = torch.min(target_q1, target_q2)
            else:
                next_act, _ = actor.sample(
                    prior,
                    solver=args.solver,
                    n_samples=256,
                    sample_steps=args.sampling_steps,
                    condition_cfg=next_obs,
                    w_cfg=1.0,
                    requires_grad=False,
                )

                with torch.no_grad():
                    target_q = torch.min(*critic_target.both(next_obs, next_act))

            target_q = (rew + (1 - tml) * 0.99 * target_q).detach()

            critic_td_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            critic_optim.zero_grad()
            critic_td_loss.backward()
            critic_optim.step()

            log["critic_td_loss"] += critic_td_loss.item()
            log["target_q"] += target_q.mean().item()

            # --- Actor Update ---
            actor.train()
            critic.eval()

            bc_loss = actor.loss(act, obs)

            new_act, _ = actor.sample(
                prior,
                solver=args.solver,
                n_samples=256,
                sample_steps=args.sampling_steps,
                condition_cfg=obs,
                w_cfg=1.0,
                use_ema=False,
                requires_grad=True,
            )

            with FreezeModules([critic]):
                q1_actor, q2_actor = critic.both(obs, new_act)

            if np.random.uniform() > 0.5:
                policy_q_loss = -q1_actor.mean() / q2_actor.abs().mean().detach()
            else:
                policy_q_loss = -q2_actor.mean() / q1_actor.abs().mean().detach()

            actor_loss = bc_loss + policy_q_loss * args.task.eta

            actor.manual_optimizers["diffusion"].zero_grad()
            actor_loss.backward()
            actor.manual_optimizers["diffusion"].step()

            log["bc_loss"] += bc_loss.item()
            log["policy_q_loss"] += policy_q_loss.item()

            step += 1

            # --- EMA Update ---
            if step % args.ema_update_interval == 0:
                if step >= 1000:
                    actor.ema_update()
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

            if step % args.log_interval == 0:
                log = {k: v / args.log_interval for k, v in log.items()}
                print(f"[{step}] {log}")
                log = dict.fromkeys(["bc_loss", "policy_q_loss", "critic_td_loss", "target_q"], 0.0)

            if step % args.save_interval == 0:
                actor.save(save_path / f"actor_step={step}.ckpt")
                torch.save(critic.state_dict(), save_path / f"critic_step={step}.ckpt")
                torch.save(critic_target.state_dict(), save_path / f"critic_target_step={step}.ckpt")

            if step >= args.training_steps:
                break

    elif args.mode == "inference":
        num_envs = args.num_envs
        num_episodes = args.num_episodes
        num_candidates = args.num_candidates

        critic.load_state_dict(torch.load(save_path / f"critic_step={args.ckpt}.ckpt", map_location=device))
        critic.eval()

        actor = DiscreteDiffusionSDE(
            nn_diffusion,
            nn_condition,
            ema_rate=0.995,
            diffusion_steps=args.sampling_steps,
            x_max=+1.0 * torch.ones((act_dim,)),
            x_min=-1.0 * torch.ones((act_dim,)),
        ).to(device)
        actor.load(save_path / f"actor_step={args.ckpt}.ckpt")
        actor.eval()

        env_eval = gym.vector.make(env_name, num_envs=num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((num_envs * num_candidates, act_dim))
        for i in range(num_episodes):
            obs, ep_reward, all_done, t = env_eval.reset(), 0.0, False, 0

            while not np.all(all_done) and t < 1000:
                obs = torch.tensor(normalizer.normalize(obs), dtype=torch.float32, device=f"cuda:{args.device_id}")
                obs = einops.repeat(obs, "b d -> (b k) d", k=num_candidates)

                act, log = actor.sample(
                    prior,
                    solver=args.solver,
                    n_samples=num_envs * num_candidates,
                    sample_steps=args.sampling_steps,
                    condition_cfg=obs,
                    w_cfg=1.0,
                )

                with torch.no_grad():
                    q = critic(obs, act)
                    act = einops.rearrange(act, "(b k) d -> b k d", k=num_candidates)
                    q = einops.rearrange(q, "(b k) 1 -> b k 1", k=num_candidates)
                    w = torch.softmax(q * args.task.weight_temperature, dim=1)

                    idx = torch.multinomial(w.squeeze(-1), num_samples=1).squeeze(-1)
                    act = act[torch.arange(num_envs), idx].cpu().numpy()

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

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards).mean(-1) * 100.0
        print(f"Score: {episode_rewards.mean():.3f}±{episode_rewards.std():.3f}")

        env_eval.close()


if __name__ == "__main__":
    pipeline()
