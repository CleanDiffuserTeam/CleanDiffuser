import os
from copy import deepcopy

import d4rl
import gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import report_parameters, FreezeModules
from utils import set_seed


class SynthERD4RLMuJoCoTDDataset(D4RLMuJoCoTDDataset):
    def __init__(self, save_path, dataset, normalize_reward: bool = False):
        super().__init__(dataset, normalize_reward)

        observations, actions, next_observations, rewards, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.float32),
            dataset["next_observations"].astype(np.float32),
            dataset["rewards"].astype(np.float32)[:, None],
            dataset["terminals"].astype(np.float32)[:, None])

        extra_transitions = np.load(save_path + "extra_transitions.npy")
        extra_observations = extra_transitions[:, :self.o_dim]
        extra_actions = extra_transitions[:, self.o_dim:self.o_dim + self.a_dim].clip(-1., 1.)
        extra_rewards = extra_transitions[:, self.o_dim + self.a_dim]
        extra_next_observations = extra_transitions[:, self.o_dim + self.a_dim + 1:self.o_dim * 2 + self.a_dim + 1]
        extra_terminals = (extra_transitions[:, -1] > 0.5).astype(np.float32)

        actions = np.concatenate([actions, extra_actions], 0)
        rewards = np.concatenate([rewards, extra_rewards[:, None]], 0)
        terminals = np.concatenate([terminals, extra_terminals[:, None]], 0)

        # Since synthER generates normalized observations, we do not need to normalize them.
        normed_observations = np.concatenate([
            self.normalizers["state"].normalize(observations), extra_observations], 0)
        normed_next_observations = np.concatenate([
            self.normalizers["state"].normalize(next_observations), extra_next_observations], 0)

        self.obs = torch.tensor(normed_observations)
        self.act = torch.tensor(actions)
        self.rew = torch.tensor(rewards)
        self.tml = torch.tensor(terminals)
        self.next_obs = torch.tensor(normed_next_observations)

        self.size = self.obs.shape[0]


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, act_dim), nn.Tanh(), )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: str = "cpu") -> np.ndarray:
        obs = torch.tensor(obs.reshape(1, -1), device=device, dtype=torch.float32)
        return self(obs).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 1), )
        self.Q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 1), )

    def both(self, obs, act):
        q1, q2 = self.Q1(torch.cat([obs, act], -1)), self.Q2(torch.cat([obs, act], -1))
        return q1, q2

    def forward(self, obs, act):
        return torch.min(*self.both(obs, act))


class TD3BC:
    def __init__(
            self, obs_dim: int, act_dim: int,
            policy_noise: float = 0.2, noise_clip: float = 0.5,
            policy_freq: int = 2, alpha: float = 2.5,
            device: str = "cpu"):

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.actor_target = deepcopy(self.actor).requires_grad_(False).eval().to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = deepcopy(self.critic).requires_grad_(False).eval().to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optim, T_max=1000_000)
        self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optim, T_max=1000_000)

        self.policy_noise, self.noise_clip, self.policy_freq, self.alpha = (
            policy_noise, noise_clip, policy_freq, alpha)

        self.device = device

    def ema_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * 0.995 + param.data * 0.005)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * 0.995 + param.data * 0.005)

    def update(self, obs, act, rew, next_obs, tml, update_actor: bool = False):

        log = {}

        with torch.no_grad():

            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_target(next_obs) + noise).clamp(-1., 1.)

            target_q = rew + (1. - tml) * 0.99 * self.critic_target(next_obs, next_act)

        current_q1, current_q2 = self.critic.both(obs, act)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.critic_lr_scheduler.step()

        log["max_target_q"] = target_q.max().item()
        log["min_target_q"] = target_q.min().item()
        log["mean_target_q"] = target_q.mean().item()
        log["critic_loss"] = critic_loss.item()

        if update_actor:

            with FreezeModules([self.critic, ]):
                pred_act = self.actor(obs)
                q = self.critic(obs, pred_act)
                lmbda = self.alpha / q.abs().mean().detach()

            policy_loss = -lmbda * q.mean()
            bc_loss = F.mse_loss(pred_act, act)

            actor_loss = policy_loss + bc_loss

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.actor_lr_scheduler.step()

            log["policy_loss"] = policy_loss.item()
            log["policy_q"] = q.mean().item()
            log["bc_loss"] = bc_loss.item()

            self.ema_update()

        else:
            log["policy_loss"] = 0.
            log["policy_q"] = 0.
            log["bc_loss"] = 0.

        return log

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])


@hydra.main(config_path="../configs/synther/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):
    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), args.normalize_reward)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = IDQLMlp(
        0, obs_dim * 2 + act_dim + 2, emb_dim=128,
        hidden_dim=1024, n_blocks=6,
        timestep_emb_type="positional")

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # --------------- Diffusion Model Actor --------------------
    synther = DiscreteDiffusionSDE(
        nn_diffusion, predict_noise=args.predict_noise, optim_params={"lr": args.diffusion_learning_rate},
        diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)

    # ---------------------- Diffusion Training ----------------------
    if args.mode == "train_diffusion":

        lr_scheduler = CosineAnnealingLR(synther.optimizer, T_max=args.diffusion_gradient_steps)

        synther.train()

        n_gradient_step = 0
        log = {"avg_diffusion_loss": 0.}

        for batch in loop_dataloader(dataloader):

            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            x = torch.cat([obs, act, rew, next_obs, tml], -1)

            log["avg_diffusion_loss"] += synther.update(x)["loss"]
            lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_diffusion_loss"] /= args.log_interval
                print(log)
                log = {"avg_diffusion_loss": 0., "gradient_steps": 0}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                synther.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                synther.save(save_path + f"diffusion_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    # ---------------------- Dataset Upsampling ----------------------------
    elif args.mode == "dataset_upsampling":

        synther.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        synther.eval()

        ori_size = dataset.obs.shape[0]
        syn_size = 5000000 - ori_size

        extra_transitions = []
        prior = torch.zeros((100000, 2 * obs_dim + act_dim + 2)).to(args.device)
        for _ in tqdm(range(syn_size // 100000)):
            syn_transitions, _ = synther.sample(
                prior, solver=args.solver, n_samples=100000, sample_steps=args.sampling_steps, use_ema=args.use_ema)
            extra_transitions.append(syn_transitions.cpu().numpy())

        syn_transitions, _ = synther.sample(
            torch.zeros((syn_size % 100000, 2 * obs_dim + act_dim + 2)).to(args.device),
            n_samples=syn_size % 100000, sample_steps=args.sampling_steps, use_ema=args.use_ema, solver=args.solver)
        extra_transitions.append(syn_transitions.cpu().numpy())
        extra_transitions = np.concatenate(extra_transitions, 0)

        np.save(save_path + "extra_transitions.npy", extra_transitions)
        print(f'Finish.')

    # --------------------- Train RL ------------------------
    elif args.mode == "train_rl":

        dataset = SynthERD4RLMuJoCoTDDataset(save_path, d4rl.qlearning_dataset(env), args.normalize_reward)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        td3bc = TD3BC(obs_dim, act_dim, device=args.device)

        n_gradient_step = 0
        log = {"avg_policy_loss": 0., "avg_bc_loss": 0., "avg_policy_q": 0., "avg_critic_loss": 0., "target_q": 0.}

        for batch in loop_dataloader(dataloader):

            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            _log = td3bc.update(obs, act, rew, next_obs, tml, bool(n_gradient_step % td3bc.policy_freq))

            log["avg_policy_loss"] += _log["policy_loss"]
            log["avg_bc_loss"] += _log["bc_loss"]
            log["avg_policy_q"] += _log["policy_q"]
            log["avg_critic_loss"] += _log["critic_loss"]
            log["target_q"] += _log["mean_target_q"]

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_policy_loss"] /= args.log_interval
                log["avg_bc_loss"] /= args.log_interval
                log["avg_policy_q"] /= args.log_interval
                log["avg_critic_loss"] /= args.log_interval
                log["target_q"] /= args.log_interval
                print(log)
                log = {"avg_policy_loss": 0., "avg_bc_loss": 0., "avg_policy_q": 0.,
                       "avg_critic_loss": 0., "target_q": 0.}

            if (n_gradient_step + 1) % args.save_interval == 0:
                td3bc.save(save_path + f"td3bc_ckpt_{n_gradient_step + 1}.pt")
                td3bc.save(save_path + f"td3bc_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step > args.rl_gradient_steps:
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":

        td3bc = TD3BC(obs_dim, act_dim, device=args.device)
        td3bc.load(save_path + f"td3bc_ckpt_{args.ckpt}.pt")
        td3bc.actor.eval()
        normalizer = dataset.normalizers["state"]

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        episode_rewards = []

        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                # sample actions
                with torch.no_grad():
                    act = td3bc.actor(obs).cpu().numpy()

                # step
                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

                if np.all(cum_done):
                    break

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
