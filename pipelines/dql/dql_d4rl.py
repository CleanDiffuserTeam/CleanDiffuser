"""
WARNING: This pipeline has not been fully tested. The results may not be accurate.
You may tune the hyperparameters in the config file before using it.
"""

import argparse
from copy import deepcopy
from pathlib import Path

import d4rl
import einops
import gym
import numpy as np
import torch
from termcolor import cprint
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import FreezeModules, loop_dataloader, set_seed
from cleandiffuser.utils.valuefuncs.iql import Qfuncs

# -- config --
argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--env_name", type=str, default="halfcheetah-medium-expert-v2")
argparser.add_argument("--mode", type=str, default="training")
argparser.add_argument("--device", type=int, default=0)
argparser.add_argument("--training_steps", type=int, default=1000000)
argparser.add_argument("--save_every_n_steps", type=int, default=200000)
argparser.add_argument("--actor_ckpt_file", type=str, default="actor-step=1000000.ckpt")
argparser.add_argument("--critic_ckpt_file", type=str, default="critic-step=1000000.ckpt")
argparser.add_argument("--sampling_steps", type=int, default=5)
argparser.add_argument("--num_envs", type=int, default=50)
argparser.add_argument("--num_episodes", type=int, default=1)
argparser.add_argument("--num_candidates", type=int, default=50)
argparser.add_argument("--ema_update_interval", type=int, default=5)
argparser.add_argument("--log_interval", type=int, default=1000)
args = argparser.parse_args()

seed = args.seed
env_name = args.env_name
mode = args.mode
device = args.device
training_steps = args.training_steps
save_every_n_steps = args.save_every_n_steps
actor_ckpt_file = args.actor_ckpt_file
critic_ckpt_file = args.critic_ckpt_file
sampling_steps = args.sampling_steps
num_envs = args.num_envs
num_episodes = args.num_episodes
num_candidates = args.num_candidates
ema_update_interval = args.ema_update_interval
log_interval = args.log_interval

# seed = 0
# env_name = "halfcheetah-medium-expert-v2"
# mode = "training"
# device = 1
# training_steps = 1000000
# save_every_n_steps = 200000
# ckpt_file = "diffusion_bc-step=1000000.ckpt"
# sampling_steps = 5
# num_envs = 50
# num_episodes = 1
# num_candidates = 50
# ema_update_interval = 5
# log_interval = 100

if env_name == "halfcheetah-medium-expert-v2":
    eta = 1.0
    weight_temperature = 50.0
elif env_name == "halfcheetah-medium-v2":
    eta = 1.0
    weight_temperature = 50.0
elif env_name == "halfcheetah-medium-replay-v2":
    eta = 1.0
    weight_temperature = 50.0
elif env_name == "hopper-medium-expert-v2":
    eta = 1.0
    weight_temperature = 1.0
elif env_name == "hopper-medium-v2":
    eta = 1.0
    weight_temperature = 300.0
elif env_name == "hopper-medium-replay-v2":
    eta = 1.0
    weight_temperature = 300.0
elif env_name == "walker2d-medium-expert-v2":
    eta = 1.0
elif env_name == "walker2d-medium-v2":
    eta = 1.0
elif env_name == "walker2d-medium-replay-v2":
    eta = 1.0
elif env_name == "kitchen-mixed-v0":
    eta = 0.005
    weight_temperature = 3.0
elif env_name == "kitchen-partial-v0":
    eta = 0.005
    weight_temperature = 10.0
elif env_name == "antmaze-medium-play-v2":
    eta = 2.0
elif env_name == "antmaze-medium-diverse-v2":
    eta = 3.0
elif env_name == "antmaze-large-play-v2":
    eta = 4.5
    weight_temperature = 5.0
elif env_name == "antmaze-large-diverse-v2":
    eta = 3.5
else:
    raise NotImplementedError(f"Env {env_name} is not supported.")

if __name__ == "__main__":
    set_seed(seed)
    device = f"cuda:{device}"

    save_path = Path(__file__).parents[2] / f"results/dql/{env_name}/"

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
    nn_diffusion = IDQLMlp(x_dim=act_dim, dropout=0.0, timestep_emb_type="positional")
    nn_condition = MLPCondition(in_dim=obs_dim, out_dim=64, hidden_dims=64, dropout=0.0)

    actor = DiscreteDiffusionSDE(
        nn_diffusion,
        nn_condition,
        ema_rate=0.995,
        diffusion_steps=sampling_steps,
        x_max=+1.0 * torch.ones((act_dim,)),
        x_min=-1.0 * torch.ones((act_dim,)),
    )

    critic = Qfuncs(obs_dim, act_dim, hidden_dim=256, n_ensembles=2)
    critic_target = deepcopy(critic).requires_grad_(False).eval()

    # --- Training ---
    if mode == "training":
        """
        Unlike other pipelines, DQL training contains two coupled training steps:
        (1) policy Q function update, and (2) diffusion BC with Q maximization.
        So it is not straightforward to use the standard pytorch-lightning trainer.
        We can instead manually train the diffusion model and the policy Q function,
        which is much more readable and convenient.
        """

        save_path.mkdir(exist_ok=True, parents=True)

        dataloader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            drop_last=True,
        )

        critic = critic.to(device)
        critic_target = critic_target.to(device)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)

        actor = actor.to(device)
        actor.configure_manual_optimizers()

        step = 0
        log = dict.fromkeys(["bc_loss", "policy_q_loss", "critic_td_loss", "target_q"], 0.0)

        prior = torch.zeros((256, act_dim), device=device)

        cprint(f"Training begin for {env_name}!", "green")
        for batch in loop_dataloader(dataloader):
            obs, next_obs = batch["obs"]["state"].to(device), batch["next_obs"]["state"].to(device)
            act = batch["act"].to(device)
            rew = batch["rew"].to(device)
            tml = batch["tml"].to(device)

            # --- Critic Update ---
            actor.eval()
            critic.train()

            q = critic(obs, act)  # (b, m, 1)

            """ Use max-Q backup for AntMaze, otherwise use policy-Q backup. """
            if "antmaze" in env_name:
                repeat_next_obs = einops.repeat(next_obs, "b d -> (b n) d", n=10)
                next_act, _ = actor.sample(
                    prior.repeat(10, 1),
                    solver="ddpm",
                    sample_steps=sampling_steps,
                    condition_cfg=repeat_next_obs,
                    w_cfg=1.0,
                    # sample_step_schedule="quad",
                    requires_grad=False,
                )

                with torch.no_grad():
                    target_q = critic_target(repeat_next_obs, next_act)  # ((b n), m, 1)
                    target_q = einops.rearrange(target_q, "(b n) m 1 -> b n m 1", n=10).max(1)[0]
                    target_q = target_q.mean(1)  # (b, 1)
            else:
                next_act, _ = actor.sample(
                    prior,
                    solver="ddpm",
                    sample_steps=sampling_steps,
                    condition_cfg=next_obs,
                    w_cfg=1.0,
                    # sample_step_schedule="quad",
                    requires_grad=False,
                )

                with torch.no_grad():
                    target_q = critic_target(next_obs, next_act).mean(1)  # (b, 1)

            target_q = (rew + (1 - tml) * 0.99 * target_q).detach()

            critic_td_loss = (q - target_q[:, None]).pow(2).mean()

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
                solver="ddpm",
                sample_steps=sampling_steps,
                condition_cfg=obs,
                w_cfg=1.0,
                # sample_step_schedule="quad",
                use_ema=False,
                requires_grad=True,
            )

            with FreezeModules([critic]):
                q_actor = critic(obs, new_act)
                q1_actor, q2_actor = q_actor[:, 0], q_actor[:, 1]

            if np.random.uniform() > 0.5:
                policy_q_loss = -q1_actor.mean() / q2_actor.abs().mean().detach()
            else:
                policy_q_loss = -q2_actor.mean() / q1_actor.abs().mean().detach()

            actor_loss = bc_loss + policy_q_loss * eta

            actor.manual_optimizers["diffusion"].zero_grad()
            actor_loss.backward()
            actor.manual_optimizers["diffusion"].step()

            log["bc_loss"] += bc_loss.item()
            log["policy_q_loss"] += policy_q_loss.item()

            step += 1

            # --- EMA Update ---
            if step % ema_update_interval == 0:
                if step >= 1000:
                    actor.ema_update()
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    """ 
                    Note: The ema ratio is set incorrectly here. However, this low ratio (0.005) actually works much better.
                    If we set it back to 0.995, the performance degrades. I think it's because the offline data distribution is stable,
                    so we don't need to overly slow down the ema update.
                    """
                    target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

            if step % log_interval == 0:
                log = {k: v / log_interval for k, v in log.items()}
                print(f"[{step}] {log}")
                log = dict.fromkeys(["bc_loss", "policy_q_loss", "critic_td_loss", "target_q"], 0.0)

            if step % save_every_n_steps == 0:
                actor.save(save_path / f"actor-step={step}.ckpt")
                torch.save(critic.state_dict(), save_path / f"critic-step={step}.ckpt")
                torch.save(
                    critic_target.state_dict(), save_path / f"critic_target-step={step}.ckpt"
                )

            if step >= training_steps:
                break

    elif mode == "inference":
        critic = critic.to(device)
        critic.load_state_dict(torch.load(save_path / critic_ckpt_file, map_location=device))
        critic.eval()

        actor = actor.to(device)
        actor.load(save_path / actor_ckpt_file)
        actor.eval()

        env_eval = gym.vector.make(env_name, num_envs=num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((num_envs * num_candidates, act_dim))
        for i in range(num_episodes):
            obs, ep_reward, all_done, t = env_eval.reset(), 0.0, False, 0

            while not np.all(all_done) and t < 1000:
                obs = torch.tensor(normalizer.normalize(obs), dtype=torch.float32, device=device)
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
                    q = critic.min_q(obs, act)
                    act = einops.rearrange(act, "(b k) d -> b k d", k=num_candidates)
                    q = einops.rearrange(q, "(b k) 1 -> b k 1", k=num_candidates)
                    w = torch.softmax(q * weight_temperature, dim=1)

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

        episode_rewards = [
            list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards
        ]
        episode_rewards = np.array(episode_rewards).mean(-1) * 100.0
        print(f"Score: {episode_rewards.mean():.3f}±{episode_rewards.std():.3f}")

        env_eval.close()
