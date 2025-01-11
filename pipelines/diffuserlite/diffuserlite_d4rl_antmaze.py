import os

import d4rl
import gym
import h5py
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cleandiffuser.dataset.d4rl_antmaze_dataset import MultiHorizonD4RLAntmazeDataset, D4RLAntmazeTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader, dict_apply
from cleandiffuser.diffusion import ContinuousRectifiedFlow
from cleandiffuser.invdynamic import FancyMlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import set_seed
from cleandiffuser.utils.iql import IQL


class MultiHorizonD4RLAntmazeDatasetwQ(MultiHorizonD4RLAntmazeDataset):
    pred_values = None

    @torch.no_grad()
    def add_value(self, iql: IQL, device: str):
        self.pred_values = np.zeros_like(self.seq_rew)
        for i in tqdm(range(self.pred_values.shape[0])):
            self.pred_values[i] = iql.V(
                torch.tensor(self.seq_obs[i], device=device)).cpu().numpy()

    def __getitem__(self, idx: int):

        indices = [
            int(self.len_each_horizon[i] * (idx / self.len_each_horizon[-1])) for i in range(len(self.horizons))]

        torch_datas = []

        for i, horizon in enumerate(self.horizons):
            path_idx, start, end = self.indices[i][indices[i]]

            rewards = self.seq_rew[path_idx, start:]
            values = (rewards * self.discount[:rewards.shape[0], None]).sum(0)

            data = {
                'obs': {
                    'state': self.seq_obs[path_idx, start:end]},
                'act': self.seq_act[path_idx, start:end],
                'rew': self.seq_rew[path_idx, start:end],
                'pred_val': self.pred_values[path_idx, start:end],
                'val': values}

            torch_data = dict_apply(data, torch.tensor)

            torch_datas.append({
                "horizon": horizon,
                "data": torch_data,
            })

        return torch_datas


@hydra.main(config_path="../configs/diffuserlite/antmaze", config_name="antmaze", version_base=None)
def pipeline(args):
    set_seed(args.seed)
    if args.test_model == "R2":
        args.diffusion_ckpt = args.reflow_ckpt

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    w_cfgs = [1.0, 0.0, 0.0]
    planning_horizons = [5, 5, 9]
    # ========================== Level Setup ==========================
    n_levels = len(planning_horizons)
    temporal_horizons = [planning_horizons[-1] for _ in range(n_levels)]
    for i in range(n_levels - 1):
        temporal_horizons[-2 - i] = (planning_horizons[-2 - i] - 1) * (temporal_horizons[-1 - i] - 1) + 1

    env = gym.make(args.task.env_name)
    dataset = MultiHorizonD4RLAntmazeDatasetwQ(
        env.get_dataset(), horizons=temporal_horizons, discount=args.discount)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # =========================== Model Setup ==========================
    fix_masks = [torch.zeros((h, obs_dim)) for h in planning_horizons]
    loss_weights = [torch.ones((h, obs_dim)) for h in planning_horizons]
    for i in range(n_levels):
        fix_idx = 0 if i == 0 else [0, -1]
        fix_masks[i][fix_idx, :] = 1.
        loss_weights[i][1, :] = args.next_obs_loss_weight

    nn_diffusions = [
        DiT1d(obs_dim, emb_dim=args.emb_dim,
              d_model=args.d_model, n_heads=args.n_heads, depth=args.depth, timestep_emb_type="fourier")
        for _ in range(n_levels)]
    nn_conditions = [
        MLPCondition(1, args.emb_dim, hidden_dims=[args.emb_dim, ])
        for _ in range(n_levels)]

    diffusions = [
        ContinuousRectifiedFlow(
            nn_diffusions[i], nn_conditions[i], fix_masks[i], loss_weights[i],
            ema_rate=args.ema_rate, device=args.device)
        for i in range(n_levels)]

    invdyn = FancyMlpInvDynamic(obs_dim, act_dim, 256, nn.Tanh(), add_dropout=True, device=args.device)

    if args.mode == "iql_training":

        dataset = D4RLAntmazeTDDataset(d4rl.qlearning_dataset(env))
        td_dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)

        iql = IQL(obs_dim, act_dim, hidden_dim=512, discount=args.discount, tau=0.7).to(args.device)
        iql.train()

        n_gradient_step = 0
        log = dict.fromkeys(["loss_v", "loss_q"], 0.)

        for batch in loop_dataloader(td_dataloader):

            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            log["loss_v"] += iql.update_V(obs, act)
            log["loss_q"] += iql.update_Q(obs, act, rew, next_obs, tml)
            iql.update_target()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log = {k: v / args.log_interval for k, v in log.items()}
                log["gradient_steps"] = n_gradient_step + 1
                print(log)
                log = dict.fromkeys(["loss_v", "loss_q"], 0.)

            if (n_gradient_step + 1) % args.save_interval == 0:
                iql.save(save_path + f'iql_ckpt_{n_gradient_step + 1}.pt')
                iql.save(save_path + f'iql_ckpt_latest.pt')

            n_gradient_step += 1
            if n_gradient_step > 1_000_000:
                break

    elif args.mode == "training":

        iql = IQL(obs_dim, act_dim, hidden_dim=512).to(args.device)
        iql.load(save_path + 'iql_ckpt_latest.pt', device=args.device)
        iql.eval()

        dataset.add_value(iql, args.device)

        lr_schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(diffusions[i].optimizer, args.diffusion_gradient_steps)
            for i in range(n_levels)]

        for diffusion in diffusions:
            diffusion.train()
        invdyn.train()

        n_gradient_step = 0
        log = dict.fromkeys([f"loss{i}" for i in range(n_levels)] + ["invdyn_loss"], 0.)
        disc_tensor = args.discount ** torch.arange(temporal_horizons[0], device=args.device)[None, :, None].float()
        for batch in loop_dataloader(dataloader):
            for i in range(n_levels):

                batch_data = batch[i]["data"]

                obs = batch_data["obs"]["state"][:, ::(temporal_horizons[i + 1] - 1) if i < n_levels - 1 else 1].to(
                    args.device)
                act = batch_data["act"][:, ::(temporal_horizons[i + 1] - 1) if i < n_levels - 1 else 1].to(args.device)
                rew = batch[i]["data"]["rew"].to(args.device)
                pred_val = batch[i]["data"]["pred_val"].to(args.device)

                rew += 1
                cum_rew = rew.cumsum(1)
                mask = (cum_rew == 0.).float()
                old_mask = mask.clone()
                mask[:, 1:] = old_mask[:, :-1]
                mask[:, 0] = 1.
                pick = (mask - old_mask).bool()
                pick[:, -1] += ~pick.any(1)
                val = rew.max(1).values / mask.sum(1)
                if i == 0:
                    val = rew - 1.
                    val[:, -1] = pred_val[:, -1]
                    val = (disc_tensor * val * mask).sum(1) / 100. + 1

                log[f"loss{i}"] += diffusions[i].update(obs, val)["loss"]
                lr_schedulers[i].step()

                if i == n_levels - 1 and n_gradient_step < args.invdyn_gradient_steps:
                    log[f"invdyn_loss"] += invdyn.update(obs[:, :-1], act[:, :-1], obs[:, 1:])["loss"]

            if (n_gradient_step + 1) % args.log_interval == 0:
                log = {k: v / args.log_interval for k, v in log.items()}
                log["gradient_steps"] = n_gradient_step + 1
                print(log)
                log = dict.fromkeys([f"loss{i}" for i in range(n_levels)] + ["invdyn_loss"], 0.)

            if (n_gradient_step + 1) % args.save_interval == 0:
                for i in range(n_levels):
                    diffusions[i].save(save_path + f'diffusion{i}_ckpt_{n_gradient_step + 1}.pt')
                    diffusions[i].save(save_path + f'diffusion{i}_ckpt_latest.pt')
                if n_gradient_step < args.invdyn_gradient_steps:
                    invdyn.save(save_path + f'invdyn_ckpt_{n_gradient_step + 1}.pt')
                    invdyn.save(save_path + f'invdyn_ckpt_latest.pt')

            n_gradient_step += 1
            if n_gradient_step > args.diffusion_gradient_steps:
                break

    elif args.mode == "prepare_dataset":

        iql = IQL(obs_dim, act_dim, hidden_dim=512).to(args.device)
        iql.load(save_path + 'iql_ckpt_latest.pt', device=args.device)
        iql.eval()
        dataset.add_value(iql, args.device)

        traj_cond_dataset, condition_dataset, traj_uncond_dataset = [], [], []
        priors = []
        ptr1, ptr2 = 0, 0
        for i in range(n_levels):
            diffusions[i].load(save_path + f'diffusion{i}_ckpt_{args.reflow_backbone_ckpt}.pt')
            diffusions[i].eval()
            traj_cond_dataset.append(
                torch.zeros((args.cond_dataset_size, 2, planning_horizons[i], obs_dim), device=args.device))
            condition_dataset.append(
                torch.zeros((args.cond_dataset_size, 1), device=args.device))
            traj_uncond_dataset.append(
                torch.zeros((args.uncond_dataset_size, 2, planning_horizons[i], obs_dim), device=args.device))
            priors.append(
                torch.zeros((args.dataset_prepare_batch_size, planning_horizons[i], obs_dim), device=args.device))

        dataloader = DataLoader(dataset, batch_size=args.dataset_prepare_batch_size, drop_last=True)
        disc_tensor = args.discount ** torch.arange(temporal_horizons[0], device=args.device)[None, :, None].float()
        for batch in loop_dataloader(dataloader):
            for i in range(n_levels):

                obs = batch[i]["data"]["obs"]["state"][:,
                      ::(temporal_horizons[i + 1] - 1) if i < n_levels - 1 else 1].to(args.device)
                rew = batch[i]["data"]["rew"].to(args.device)
                pred_val = batch[i]["data"]["pred_val"].to(args.device)

                rew += 1
                cum_rew = rew.cumsum(1)
                mask = (cum_rew == 0.).float()
                old_mask = mask.clone()
                mask[:, 1:] = old_mask[:, :-1]
                mask[:, 0] = 1.
                pick = (mask - old_mask).bool()
                pick[:, -1] += ~pick.any(1)
                val = rew.max(1).values / mask.sum(1)
                if i == 0:
                    val = rew - 1.
                    val[:, -1] = pred_val[:, -1]
                    val = (disc_tensor * val * mask).sum(1) / 100. + 1

                if i == 0:
                    priors[i][:, 0] = obs[:, 0]
                else:
                    priors[i][:, [0, -1]] = obs[:, [0, -1]]

                if ptr1 < args.cond_dataset_size:
                    noise_cond = torch.randn_like(priors[i])
                    traj_cond, _ = diffusions[i].sample(
                        priors[i], x1=noise_cond, n_samples=obs.shape[0],
                        sample_steps=args.dataset_prepare_sampling_steps, use_ema=True,
                        condition_cfg=val,
                        w_cfg=1.0, temperature=1.0,
                        sample_step_schedule="quad_continuous")
                    traj_cond_dataset[i][ptr1:ptr1 + obs.shape[0], 0] = traj_cond
                    traj_cond_dataset[i][ptr1:ptr1 + obs.shape[0], 1] = noise_cond
                    condition_dataset[i][ptr1:ptr1 + obs.shape[0]] = val

                if ptr2 < args.uncond_dataset_size:
                    noise_uncond = torch.randn_like(priors[i])
                    traj_uncond, _ = diffusions[i].sample(
                        priors[i], x1=noise_uncond, n_samples=obs.shape[0],
                        sample_steps=args.dataset_prepare_sampling_steps, use_ema=True,
                        condition_cfg=None,
                        w_cfg=0.0, temperature=1.0,
                        sample_step_schedule="quad_continuous")
                    traj_uncond_dataset[i][ptr2:ptr2 + obs.shape[0], 0] = traj_uncond
                    traj_uncond_dataset[i][ptr2:ptr2 + obs.shape[0], 1] = noise_uncond

            if ptr1 < args.cond_dataset_size:
                ptr1 += obs.shape[0]
            if ptr2 < args.uncond_dataset_size:
                ptr2 += obs.shape[0]
            print(
                f'cond: {ptr1 / args.cond_dataset_size * 100.:.1f}%, '
                f'uncon: {ptr2 / args.uncond_dataset_size * 100.:.1f}%')
            if ptr1 >= args.cond_dataset_size and ptr2 >= args.uncond_dataset_size:
                break

        with h5py.File(save_path + "traj_cond_dataset.h5", "w") as f:
            for i in range(n_levels):
                f.create_dataset(f"traj_cond_dataset_{i}", data=traj_cond_dataset[i].cpu().numpy())
                f.create_dataset(f"condition_dataset_{i}", data=condition_dataset[i].cpu().numpy())
        with h5py.File(save_path + "traj_uncond_dataset.h5", "w") as f:
            for i in range(n_levels):
                f.create_dataset(f"traj_uncond_dataset_{i}", data=traj_uncond_dataset[i].cpu().numpy())

    elif args.mode == "reflow":

        for i in range(n_levels):
            diffusions[i].load(save_path + f'diffusion{i}_ckpt_{args.reflow_backbone_ckpt}.pt')
            diffusions[i].train()
            diffusions[i].optimizer.learning_rate = 2e-5

        traj_cond_dataset, traj_uncond_dataset = [], []
        condition_dataset = []
        with h5py.File(save_path + "traj_cond_dataset.h5", "r") as f:
            for i in range(n_levels):
                traj_cond_dataset.append(torch.tensor(f[f"traj_cond_dataset_{i}"][:], device=args.device))
                condition_dataset.append(torch.tensor(f[f"condition_dataset_{i}"][:], device=args.device))
        with h5py.File(save_path + "traj_uncond_dataset.h5", "r") as f:
            for i in range(n_levels):
                traj_uncond_dataset.append(torch.tensor(f[f"traj_uncond_dataset_{i}"][:], device=args.device))

        log = dict.fromkeys([f"loss{i}" for i in range(n_levels)], 0.)
        for n_gradient_step in range(args.reflow_gradient_steps):
            for i in range(n_levels):
                if (n_gradient_step % 5) == 0:
                    idx = torch.randint(args.uncond_dataset_size, (args.batch_size,), device=args.device)
                    x01 = traj_uncond_dataset[i][idx]
                    x0, x1 = x01[:, 0], x01[:, 1]
                    log[f"loss{i}"] += diffusions[i].update(x0, x1=x1)["loss"]
                else:
                    idx = torch.randint(args.cond_dataset_size, (args.batch_size,), device=args.device)
                    x01 = traj_cond_dataset[i][idx]
                    val = condition_dataset[i][idx]
                    x0, x1 = x01[:, 0], x01[:, 1]
                    log[f"loss{i}"] += diffusions[i].update(x0, val, x1=x1)["loss"]

            if (n_gradient_step + 1) % args.log_interval == 0:
                log = {k: v / args.log_interval for k, v in log.items()}
                log["gradient_steps"] = n_gradient_step + 1
                print(log)
                log = dict.fromkeys([f"loss{i}" for i in range(n_levels)], 0.)

            if (n_gradient_step + 1) % args.save_interval == 0:
                for i in range(n_levels):
                    diffusions[i].save(save_path + f'reflow{i}_ckpt_{n_gradient_step + 1}.pt')
                    diffusions[i].save(save_path + f'reflow{i}_ckpt_latest.pt')

    elif args.mode == "inference":

        n_candidates = args.num_candidates

        for i in range(n_levels):
            diffusions[i].load(
                save_path + f'{"reflow" if args.test_model == "R2" else "diffusion"}{i}_ckpt_{args.diffusion_ckpt}.pt')
            diffusions[i].eval()
        invdyn.load(save_path + f'invdyn_ckpt_{args.invdyn_ckpt}.pt')
        invdyn.eval()

        iql = IQL(obs_dim, act_dim, hidden_dim=512).to(args.device)
        iql.load(save_path + 'iql_ckpt_latest.pt', device=args.device)
        iql.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        priors = [torch.zeros((args.num_envs, planning_horizons[i], obs_dim),
                              device=args.device) for i in range(n_levels)]
        priors[0] = torch.zeros((args.num_envs, n_candidates, planning_horizons[0], obs_dim)).to(args.device)
        condition = torch.ones((args.num_envs, n_candidates, 1), device=args.device)
        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:

                target_return = np.ones(args.num_envs, dtype=np.float32)
                if "medium-play" in args.task.env_name:
                    target_return[:] = 0.2
                    target_return[obs[:, 1] > 18.] = 0.8
                elif "medium-diverse" in args.task.env_name:
                    target_return[:] = 0.2
                    target_return[obs[:, 0] > 10.] = 0.3
                    target_return[obs[:, 1] > 15.] = 0.8
                elif "large-play" in args.task.env_name:
                    target_return[:] = 0.6
                    target_return[np.logical_and(obs[:, 0] >= 13., obs[:, 1] < 28)] = 0.25
                    target_return[obs[:, 0] < 13.] = 0.1
                elif "large_diverse" in args.task.env_name:
                    target_return[:] = 0.6
                    target_return[np.logical_and(obs[:, 0] >= 13., obs[:, 1] < 28)] = 0.3
                    target_return[obs[:, 0] < 13.] = 0.25
                target_return = torch.tensor(target_return, device=args.device)[:, None, None]
                this_condition = condition * target_return

                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                priors[0][:, :, 0] = obs.unsqueeze(1)
                for j in range(n_levels):
                    traj, _ = diffusions[j].sample(
                        priors[j].view(-1, planning_horizons[j], obs_dim),
                        n_samples=args.num_envs * n_candidates if j == 0 else args.num_envs,
                        sample_steps=2 if args.test_model == "R2" else 5, use_ema=args.use_ema,
                        condition_cfg=(this_condition.reshape(-1, 1) if j == 0 else this_condition[:, 0]),
                        w_cfg=w_cfgs[j], temperature=args.temperature,
                        sample_step_schedule="quad_continuous")
                    if j == 0:
                        traj = traj.reshape(args.num_envs, n_candidates, -1, obs_dim)
                        with torch.no_grad():
                            value = iql.V(traj[:, :, 1])[:, :, 0]
                            idx = torch.argmax(value, -1)
                            traj = traj[torch.arange(args.num_envs), idx]
                    if j < n_levels - 1:
                        priors[j + 1][:, [0, -1]] = traj[:, [0, 1]]

                with torch.no_grad():
                    act = invdyn(traj[:, 0], traj[:, 1]).cpu().numpy()

                obs, rew, done, info = env_eval.step(act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += rew
                print(f'[t={t}] xy: {obs[:, :2]}')
                print(f'[t={t}] rew: {ep_reward}')

            episode_rewards.append(np.clip(ep_reward, 0., 1.))

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))

        env_eval.close()


if __name__ == "__main__":
    pipeline()
