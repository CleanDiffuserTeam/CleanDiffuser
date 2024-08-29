import os
from copy import deepcopy

import d4rl
import gym
import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import report_parameters, IDQLQNet, IDQLVNet
from utils import set_seed


@hydra.main(config_path="../configs/idql/mujoco", config_name="mujoco", version_base=None)
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
        obs_dim, act_dim, emb_dim=64,
        hidden_dim=args.actor_hidden_dim, n_blocks=args.actor_n_blocks, dropout=args.actor_dropout,
        timestep_emb_type="positional")
    nn_condition = IdentityCondition(dropout=0.0)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # --------------- Diffusion Model Actor --------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
        x_max=+1. * torch.ones((1, act_dim)),
        x_min=-1. * torch.ones((1, act_dim)),
        diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)

    # ------------------ Critic ---------------------
    iql_q = IDQLQNet(obs_dim, act_dim, hidden_dim=args.critic_hidden_dim).to(args.device)
    iql_q_target = deepcopy(iql_q).requires_grad_(False).eval()
    iql_v = IDQLVNet(obs_dim, hidden_dim=args.critic_hidden_dim).to(args.device)

    q_optim = torch.optim.Adam(iql_q.parameters(), lr=args.critic_learning_rate)
    v_optim = torch.optim.Adam(iql_v.parameters(), lr=args.critic_learning_rate)

    # ---------------------- Training ----------------------
    if args.mode == "train":

        actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
        q_lr_scheduler = CosineAnnealingLR(q_optim, T_max=args.gradient_steps)
        v_lr_scheduler = CosineAnnealingLR(v_optim, T_max=args.gradient_steps)

        actor.train()
        iql_q.train()
        iql_v.train()

        n_gradient_step = 0
        log = {"bc_loss": 0., "q_loss": 0., "v_loss": 0.}

        for batch in loop_dataloader(dataloader):

            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            # -- IQL Training
            if n_gradient_step % 2 == 0:

                q = iql_q_target(obs, act)
                v = iql_v(obs)
                v_loss = (torch.abs(args.iql_tau - ((q - v) < 0).float()) * (q - v) ** 2).mean()

                v_optim.zero_grad()
                v_loss.backward()
                v_optim.step()

                with torch.no_grad():
                    td_target = rew + args.discount * (1 - tml) * iql_v(next_obs)
                q1, q2 = iql_q.both(obs, act)
                q_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()
                q_optim.zero_grad()
                q_loss.backward()
                q_optim.step()

                q_lr_scheduler.step()
                v_lr_scheduler.step()

                for param, target_param in zip(iql_q.parameters(), iql_q_target.parameters()):
                    target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

            # -- Policy Training
            bc_loss = actor.update(act, obs)["loss"]
            actor_lr_scheduler.step()

            # # ----------- Logging ------------
            log["bc_loss"] += bc_loss
            log["q_loss"] += q_loss.item()
            log["v_loss"] += v_loss.item()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["bc_loss"] /= args.log_interval
                log["q_loss"] /= args.log_interval
                log["v_loss"] /= args.log_interval
                print(log)
                log = {"bc_loss": 0., "q_loss": 0., "v_loss": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                actor.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                actor.save(save_path + f"diffusion_ckpt_latest.pt")
                torch.save({
                    "iql_q": iql_q.state_dict(),
                    "iql_q_target": iql_q_target.state_dict(),
                    "iql_v": iql_v.state_dict(),
                }, save_path + f"iql_ckpt_{n_gradient_step + 1}.pt")
                torch.save({
                    "iql_q": iql_q.state_dict(),
                    "iql_q_target": iql_q_target.state_dict(),
                    "iql_v": iql_v.state_dict(),
                }, save_path + f"iql_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.gradient_steps:
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":

        actor.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        critic_ckpt = torch.load(save_path + f"iql_ckpt_{args.ckpt}.pt")
        iql_q.load_state_dict(critic_ckpt["iql_q"])
        iql_q_target.load_state_dict(critic_ckpt["iql_q_target"])
        iql_v.load_state_dict(critic_ckpt["iql_v"])

        actor.eval()
        iql_q.eval()
        iql_v.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
        for i in range(args.num_episodes):

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 1000 + 1:
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)

                # sample actions
                act, log = actor.sample(
                    prior,
                    solver=args.solver,
                    n_samples=args.num_envs * args.num_candidates,
                    sample_steps=args.sampling_steps,
                    condition_cfg=obs, w_cfg=1.0,
                    use_ema=args.use_ema, temperature=args.temperature)

                # resample
                with torch.no_grad():
                    q = iql_q_target(obs, act)
                    v = iql_v(obs)
                    adv = (q - v)
                    adv = adv.view(-1, args.num_candidates, 1)

                    w = torch.softmax(adv * args.weight_temperature, 1)
                    act = act.view(-1, args.num_candidates, act_dim)

                    p = w / w.sum(1, keepdim=True)

                    indices = torch.multinomial(p.squeeze(-1), 1).squeeze(-1)
                    sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

                # step
                obs, rew, done, info = env_eval.step(sampled_act)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)) if t < 1000 else rew, 2)}')

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
