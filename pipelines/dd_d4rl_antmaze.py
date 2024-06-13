import os

import d4rl
import gym
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE
from utils import set_seed


@hydra.main(config_path="../configs/dd/antmaze", config_name="antmaze", version_base=None)
def pipeline(args):

    return_scale = DD_RETURN_SCALE[args.task.env_name]

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLAntmazeDataset(
        env.get_dataset(), horizon=args.task.horizon,
        noreaching_penalty=args.noreaching_penalty, discount=args.discount)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = DiT1d(
        obs_dim, emb_dim=args.emb_dim,
        d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
    nn_condition = MLPCondition(
        in_dim=1, out_dim=args.emb_dim, hidden_dims=[args.emb_dim, ], act=nn.SiLU(), dropout=args.label_dropout)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.horizon, obs_dim))
    fix_mask[0] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim))
    loss_weight[1] = args.next_obs_loss_weight

    # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    agent = ContinuousDiffusionSDE(
        nn_diffusion, nn_condition,
        fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=args.ema_rate,
        device=args.device, predict_noise=args.predict_noise, noise_schedule="linear")

    # --------------- Inverse Dynamic -------------------
    invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)

    # ---------------------- Training ----------------------
    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)

        agent.train()
        invdyn.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0.,  "avg_loss_invdyn": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device) / return_scale + 1.  # rescale to [0, 1]

            # ----------- Gradient Step ------------
            log["avg_loss_diffusion"] += agent.update(obs, val)['loss']
            diffusion_lr_scheduler.step()
            if n_gradient_step <= args.classifier_gradient_steps:
                log["avg_loss_invdyn"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])['loss']
                invdyn_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_invdyn"] /= args.log_interval
                print(log)
                log = {"avg_loss_diffusion": 0., "avg_loss_invdyn": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                invdyn.save(save_path + f"invdyn_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")
                invdyn.save(save_path + f"invdyn_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":

        agent.load(save_path + f"diffusion_ckpt_{args.diffusion_ckpt}.pt")
        agent.eval()
        invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
        invdyn.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim), device=args.device)
        condition = torch.ones((args.num_envs, 1), device=args.device) * args.task.target_return
        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                # sample trajectories
                prior[:, 0] = obs
                traj, log = agent.sample(
                    prior, solver=args.solver,
                    n_samples=args.num_envs, sample_steps=args.sampling_steps, use_ema=args.use_ema,
                    condition_cfg=condition, w_cfg=args.task.w_cfg, temperature=args.temperature)

                # inverse dynamic
                with torch.no_grad():
                    act = invdyn.predict(obs, traj[:, 1, :]).cpu().numpy()

                # step
                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += rew
                print(f'[t={t}] xy: {np.around(obs[:, :2], 2)}')
                print(f'[t={t}] rew: {ep_reward}')

            episode_rewards.append(np.clip(ep_reward, 0., 1.))

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
