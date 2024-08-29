import os
from typing import Dict

import d4rl
import gym
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import SfBCUNet
from cleandiffuser.utils import loop_dataloader, set_seed, Mlp, dict_apply, GaussianNormalizer


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


class InsamplePlanningD4RLMuJoCoDataset(D4RLMuJoCoDataset):
    def __init__(
            self, dataset: Dict[str, np.ndarray], terminal_penalty: float = -100.,
            horizon: int = 1, max_path_length: int = 1000, discount: float = 0.99, ):
        super().__init__(dataset, terminal_penalty, horizon, max_path_length, discount)
        self.val_normalizer = GaussianNormalizer(self.seq_val)
        self.normed_seq_val = self.val_normalizer.normalize(self.seq_val)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]
        data = {
            'obs': {
                'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
            'rew': self.seq_rew[path_idx, start:end],
            'val': self.normed_seq_val[path_idx, start:end], }
        return dict_apply(data, torch.tensor)


@hydra.main(config_path="../configs/sfbc/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):
    set_seed(args.seed)
    device, env_name = args.device, args.task.env_name
    M, alpha = args.monte_carlo_samples, args.weight_temperature

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    env = gym.make(env_name)
    dataset = InsamplePlanningD4RLMuJoCoDataset(
        env.get_dataset(), horizon=32, discount=args.discount)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    n_trajs, max_traj_len = dataset.seq_obs.shape[0], dataset.seq_obs.shape[1]

    nn_diffusion = SfBCUNet(act_dim, emb_dim=64)
    nn_condition = MLPCondition(obs_dim, 64, [64, ], torch.nn.SiLU())

    actor = ContinuousDiffusionSDE(
        nn_diffusion, nn_condition, ema_rate=args.ema_rate, predict_noise=args.predict_noise,
        x_max=+1. * torch.ones((act_dim,)),
        x_min=-1. * torch.ones((act_dim,)), device=device, optim_params={"lr": args.actor_learning_rate})

    critic = Mlp(obs_dim + act_dim, [args.hidden_dim, args.hidden_dim],
                 1, torch.nn.SiLU()).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

    if args.mode == "bc_training":

        n_gradient_step = 0
        log = dict.fromkeys(["bc_loss"], 0.)
        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].reshape(-1, obs_dim).to(device)
            act = batch["act"].reshape(-1, obs_dim).to(device)

            log["bc_loss"] += actor.update(act, obs)["loss"]

            if (n_gradient_step + 1) % args.log_interval == 0:
                log = {k: v / args.log_interval for k, v in log.items()}
                log["gradient_step"] = n_gradient_step + 1
                print(log)
                log = dict.fromkeys(["bc_loss"], 0.)

            if (n_gradient_step + 1) % args.save_interval == 0:
                actor.save(save_path + f'diffusion_ckpt_{n_gradient_step + 1}.pt')

            n_gradient_step += 1
            if n_gradient_step > args.bc_gradient_steps:
                break

    elif args.mode == "critic_training":

        actor.load(save_path + f'diffusion_ckpt_{args.eval_actor_ckpt}.pt')
        actor.eval()

        for k in range(args.q_training_iters):

            if k > 0:
                critic.eval()
                # M Monte Carlo evaluation
                normed_eval_seq_val = np.empty_like(dataset.normed_seq_val)
                for i in tqdm(range(dataset.normed_seq_val.shape[0])):
                    obs = torch.tensor(dataset.seq_obs[i], device=device).unsqueeze(1).repeat(1, M, 1)
                    prior = torch.zeros((max_traj_len * M, act_dim))
                    act = actor.sample(
                        prior, solver=args.eval_actor_solver, n_samples=max_traj_len * M,
                        sample_steps=args.eval_actor_sampling_steps,
                        condition_cfg=obs.view(-1, obs_dim))[0].view(-1, M, act_dim)

                    with torch.no_grad():
                        pred_val = critic(torch.cat([obs, act], dim=-1))

                    weight = torch.nn.functional.softmax(alpha * pred_val, dim=1)
                    normed_eval_seq_val[i] = (weight * pred_val).sum(1).cpu().numpy()

                # Implicit in-sample planning
                eval_seq_val = dataset.val_normalizer.unnormalize(normed_eval_seq_val)

                target_seq_val = np.empty_like(eval_seq_val)
                target_seq_val[:, :-1] = \
                    (dataset.seq_rew[:, :-1] + args.discount * np.maximum(dataset.seq_val[:, 1:], eval_seq_val[:, 1:]))
                target_seq_val[:, -1] = eval_seq_val[:, -1]
                if dataset.tml_and_not_timeout.shape[0] != 0:
                    target_seq_val[np.where(dataset.tml_and_not_timeout)] = (
                        dataset.seq_rew)[np.where(dataset.tml_and_not_timeout)]
                dataset.seq_val = target_seq_val
                dataset.val_normalizer.__init__(target_seq_val)
                dataset.normed_seq_val = dataset.val_normalizer.normalize(target_seq_val)
                dataloader = DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
                critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

            # Critic training, reset critic for each iteration
            n_gradient_step = 0
            critic.apply(weight_init)
            critic.train()
            log = dict.fromkeys(["critic_loss"], 0.)
            for batch in loop_dataloader(dataloader):
                obs, act, val = batch["obs"]["state"].to(device), batch["act"].to(device), batch["val"].to(device)
                critic_loss = (critic(torch.cat([obs, act], dim=-1)) - val).pow(2).mean()
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()
                log["critic_loss"] += critic_loss.item()
                if (n_gradient_step + 1) % args.log_interval == 0:
                    log = {k: v / args.log_interval for k, v in log.items()}
                    log["gradient_step"] = n_gradient_step + 1
                    print(log)
                    log = dict.fromkeys(["critic_loss"], 0.)
                n_gradient_step += 1
                if n_gradient_step > args.critic_gradient_steps:
                    break

        torch.save(critic.state_dict(), save_path + "critic.pt")

    elif args.mode == "inference":

        num_candidates = args.num_candidates

        actor.load(save_path + f'diffusion_ckpt_{args.ckpt}.pt')
        actor.eval()
        critic.load_state_dict(torch.load(save_path + "critic.pt"))
        critic.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((args.num_envs, num_candidates, act_dim))

        for i in range(args.num_episodes):
            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:

                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                condition = obs[:, None, :].repeat(1, num_candidates, 1)

                act, _ = actor.sample(
                    prior.view(-1, act_dim), solver=args.solver,
                    n_samples=num_candidates * args.num_envs, sample_steps=args.sampling_steps,
                    use_ema=True, condition_cfg=condition.view(-1, obs_dim), w_cfg=1.0,
                    temperature=args.temperature)
                act = act.view(args.num_envs, num_candidates, act_dim)

                with torch.no_grad():
                    value = critic(torch.cat([condition, act], -1))[..., 0]
                    sorted_indices = torch.argsort(value, dim=1, descending=True)
                    act = act.gather(1, sorted_indices.unsqueeze(-1).expand(-1, -1, act.size(-1)))
                    act = act[:, :args.top_k_average].mean(1).cpu().numpy()

                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

            episode_rewards.append(ep_reward)

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))


if __name__ == "__main__":
    pipeline()
