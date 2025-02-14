from pathlib import Path
from typing import Union

import d4rl
import einops
import gym
import hydra
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeDataset, D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenDataset, D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset, D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import SfBCUNet
from cleandiffuser.utils import GaussianNormalizer, Mlp, dict_apply, loop_dataloader
from pytorch_lightning.loggers import WandbLogger


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


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


@hydra.main(config_path="../../configs/sfbc", config_name="d4rl", version_base=None)
def pipeline(args):
    L.seed_everything(args.seed, workers=True)
    device = f"cuda:{args.device_id}"
    
    env_name = args.task.env_name
    M, alpha = args.monte_carlo_samples, args.weight_temperature
    save_path = Path(__file__).parents[1] / f"results/{args.pipeline_name}/{env_name}/"

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

    nn_diffusion = SfBCUNet(act_dim, emb_dim=64)
    nn_condition = MLPCondition(obs_dim, 64, [64], torch.nn.SiLU())

    actor = ContinuousDiffusionSDE(
        nn_diffusion,
        nn_condition,
        ema_rate=0.999,
        x_max=+1.0 * torch.ones((act_dim,)),
        x_min=-1.0 * torch.ones((act_dim,)),
    )

    critic = Mlp(obs_dim + act_dim, [256, 256], 1, torch.nn.SiLU()).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    if args.mode == "bc_training":
        dataloader = DataLoader(
            BC_Wrapper(dataset), batch_size=512, shuffle=True, num_workers=4, persistent_workers=True
        )

        callback = ModelCheckpoint(
            dirpath=save_path, filename="diffusion_bc-{step}", every_n_train_steps=args.save_interval
        )

        wandb_logger = WandbLogger(
            project="cleandiffuser",
            config=dict(args),
            name=f"{args.pipeline_name}-{args.task.env_name}-{args.mode}"
        )

        trainer = L.Trainer(
            accelerator="gpu",
            devices=-1,
            max_steps=args.bc_training_steps,
            deterministic=True,
            log_every_n_steps=200,
            default_root_dir=save_path,
            callbacks=[callback],
            logger=wandb_logger
        )

        trainer.fit(actor, dataloader)

    elif args.mode == "critic_training":
        actor.load_state_dict(
            torch.load(save_path / f"diffusion_bc-step={args.eval_actor_ckpt}.ckpt", map_location=device)["state_dict"]
        )
        actor.to(device).eval()

        if "kitchen" in env_name:
            dataset = D4RLKitchenDataset(env.get_dataset(), horizon=32)
        elif "antmaze" in env_name:
            dataset = D4RLAntmazeDataset(env.get_dataset(), horizon=32)
        else:
            dataset = D4RLMuJoCoDataset(env.get_dataset(), horizon=32)

        dataset = InsamplePlanningD4RLDataset(dataset)

        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)

        _, max_traj_len = dataset.seq_obs.shape[0], dataset.seq_obs.shape[1]

        for k in range(args.q_training_iters):
            if k > 0:
                critic.eval()
                # M Monte Carlo evaluation
                normed_eval_seq_val = np.empty_like(dataset.normed_seq_val)
                for i in tqdm(range(dataset.normed_seq_val.shape[0])):
                    obs = torch.tensor(dataset.seq_obs[i], device=device).unsqueeze(1).repeat(1, M, 1)
                    prior = torch.zeros((max_traj_len * M, act_dim))
                    act = actor.sample(
                        prior,
                        solver=args.eval_actor_solver,
                        n_samples=max_traj_len * M,
                        sample_steps=args.eval_actor_sampling_steps,
                        condition_cfg=obs.view(-1, obs_dim),
                    )[0].view(-1, M, act_dim)

                    with torch.no_grad():
                        pred_val = critic(torch.cat([obs, act], dim=-1))

                    weight = torch.nn.functional.softmax(alpha * pred_val, dim=1)
                    normed_eval_seq_val[i] = (weight * pred_val).sum(1).cpu().numpy()

                # Implicit in-sample planning
                eval_seq_val = dataset.val_normalizer.unnormalize(normed_eval_seq_val)

                target_seq_val = np.empty_like(eval_seq_val)
                target_seq_val[:, :-1] = dataset.seq_rew[:, :-1] + args.discount * np.maximum(
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
                dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)
                critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

            # Critic training, reset critic for each iteration
            n_gradient_step = 0
            critic.apply(weight_init)
            critic.train()
            log = dict.fromkeys(["critic_loss"], 0.0)
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
                    log = dict.fromkeys(["critic_loss"], 0.0)
                n_gradient_step += 1
                if n_gradient_step > args.critic_training_steps:
                    break

        torch.save(critic.state_dict(), save_path / "critic.pt")

    elif args.mode == "inference":
        num_candidates = args.num_candidates
        num_envs = args.num_envs

        actor.load_state_dict(
            torch.load(save_path / f"diffusion_bc-step={args.eval_actor_ckpt}.ckpt", map_location=device)["state_dict"]
        )
        actor.to(device).eval()
        critic.load_state_dict(torch.load(save_path / "critic.pt", map_location=device))
        critic.eval()

        env_eval = gym.vector.make(env_name, num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((num_envs * num_candidates, act_dim))

        for i in range(args.num_episodes):
            obs, ep_reward, all_done, t = env_eval.reset(), 0.0, False, 0

            while not np.all(all_done) and t < 1000:
                obs = torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32)
                obs = einops.repeat(obs, "b d -> (b k) d", k=num_candidates)

                act, _ = actor.sample(
                    prior,
                    solver=args.solver,
                    n_samples=num_candidates * num_envs,
                    sample_steps=args.sampling_steps,
                    condition_cfg=obs,
                    w_cfg=1.0,
                )

                with torch.no_grad():
                    value = critic(torch.cat([obs, act], -1))
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

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards).mean(-1) * 100.0
        print(f"Score: {episode_rewards.mean():.3f}±{episode_rewards.std():.3f}")

        env_eval.close()


if __name__ == "__main__":
    pipeline()
