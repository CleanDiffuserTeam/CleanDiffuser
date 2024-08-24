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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import FreezeModules, loop_dataloader
from cleandiffuser.utils.iql import TwinQ


@hydra.main(config_path="../configs/dql", config_name="d4rl", version_base=None)
def pipeline(args):

    L.seed_everything(args.seed, workers=True)

    env_name = args.task.env_name
    save_path = Path(__file__).parents[1] / \
        f"results/{args.pipeline_name}/{env_name}/"

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
        x_dim=act_dim, emb_dim=64, hidden_dim=256, n_blocks=3, dropout=0.1,
        timestep_emb_type="positional")
    nn_condition = MLPCondition(
        in_dim=obs_dim, out_dim=64, hidden_dims=[64, ], act=torch.nn.SiLU(), dropout=0.0)

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
            dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)

        critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)

        """
        While a high `ema_rate` performs better when using the standard
        diffusion loss, it causes instability when adding an additional
        Q maximization loss. So we use a smaller `ema_rate` here.
        """
        actor = DiscreteDiffusionSDE(
            nn_diffusion, nn_condition, ema_rate=0.995,
            diffusion_steps=args.sampling_steps,
            x_max=+1.0 * torch.ones((act_dim,)),
            x_min=-1.0 * torch.ones((act_dim,))).to(device)

        step = 0
        log = dict.fromkeys(["bc_loss", "policy_q_loss",
                            "critic_td_loss", "target_q"], 0.0)

        prior = torch.zeros((256, act_dim), device=device)

        for batch in loop_dataloader(dataloader):

            obs, next_obs = batch["obs"]["state"].to(
                device), batch["next_obs"]["state"].to(device)
            act = batch["act"].to(device)
            rew = batch["rew"].to(device)
            tml = batch["tml"].to(device)

            # --- Critic Update ---
            actor.eval()
            critic.train()

            q1, q2 = critic.both(obs, act)

            """ Use max-Q backup for AntMaze, otherwise use policy-Q backup. """
            if "antmaze" in env_name:

                repeat_next_obs = einops.repeat(
                    next_obs, "b d -> (b n) d", n=10)
                next_act, _ = actor.sample(
                    prior.repeat(10, 1), solver=args.solver,
                    n_samples=2560, sample_steps=args.sampling_steps,
                    condition_cfg=repeat_next_obs, w_cfg=1.0,
                    requires_grad=False)

                with torch.no_grad():
                    target_q1, target_q2 = critic_target.both(
                        repeat_next_obs, next_act)
                    target_q1 = einops.rearrange(
                        target_q1, "(b n) 1 -> b n 1", n=10).max(1)[0]
                    target_q2 = einops.rearrange(
                        target_q2, "(b n) 1 -> b n 1", n=10).max(1)[0]
                    target_q = torch.min(target_q1, target_q2)
            else:
                next_act, _ = actor.sample(
                    prior, solver=args.solver,
                    n_samples=256, sample_steps=args.sampling_steps,
                    condition_cfg=next_obs, w_cfg=1.0,
                    requires_grad=False)

                with torch.no_grad():
                    target_q = torch.min(
                        *critic_target.both(next_obs, next_act))

            target_q = (rew + (1 - tml) * 0.99 * target_q)

            critic_td_loss = F.mse_loss(
                q1, target_q) + F.mse_loss(q2, target_q)

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
                prior, solver=args.solver,
                n_samples=256, sample_steps=args.sampling_steps,
                condition_cfg=obs, w_cfg=1.0,
                requires_grad=True)

            with FreezeModules([critic, ]):
                q1_actor, q2_actor = critic.both(obs, new_act)

            if step % 2 == 0:
                policy_q_loss = - q1_actor.mean() / q2_actor.abs().mean()
            else:
                policy_q_loss = - q2_actor.mean() / q1_actor.abs().mean()

            actor_loss = bc_loss + policy_q_loss * args.task.eta

            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()

            log["bc_loss"] += bc_loss.item()
            log["policy_q_loss"] += policy_q_loss.item()

            # --- EMA Update ---
            if step % args.ema_update_interval == 0:
                actor.ema_update()
                for p, p_target in zip(critic.parameters(), critic_target.parameters()):
                    p_target.data.copy_(0.995 * p_target.data + 0.005 * p.data)

            step += 1

            if step % args.log_interval == 0:
                log = {k: v / args.log_interval for k, v in log.items()}
                print(f"[{step}] {log}")
                log = dict.fromkeys(
                    ["bc_loss", "policy_q_loss", "critic_td_loss", "target_q"], 0.0)

            if step % args.save_interval == 0:
                actor.save(save_path + f"actor_step={step}.pt")
                torch.save(critic.state_dict(), save_path +
                           f"critic_step={step}.pt")
                torch.save(critic_target.state_dict(), save_path +
                           f"critic_target_step={step}.pt")

            if step >= args.training_steps:
                break


if __name__ == "__main__":
    pipeline()
