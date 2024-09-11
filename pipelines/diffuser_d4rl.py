"""
WARNING: This pipeline has not been fully tested. The results may not be accurate.
You may tune the hyperparameters in the config file before using it.
"""

from pathlib import Path

import d4rl
import einops
import gym
import hydra
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from cleandiffuser.classifier import OptimalityClassifier
from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import DiT1d

RETURN_SCALE = {
    "halfcheetah-medium-expert-v2": 1200,
    "halfcheetah-medium-replay-v2": 550,
    "halfcheetah-medium-v2": 580,
    "hopper-medium-expert-v2": 400,
    "hopper-medium-replay-v2": 340,
    "hopper-medium-v2": 350,
    "walker2d-medium-expert-v2": 550,
    "walker2d-medium-replay-v2": 470,
    "walker2d-medium-v2": 480,
    "kitchen-partial-v0": 270,
    "kitchen-mixed-v0": 220,
    "antmaze-medium-play-v2": 100,
    "antmaze-medium-diverse-v2": 100,
    "antmaze-large-play-v2": 100,
    "antmaze-large-diverse-v2": 100,
}


class ObsActSequence_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, env_name: str):
        self.env_name = env_name
        self.dataset = dataset
        self.scale = RETURN_SCALE[env_name]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        path_idx, start, end = self.indices[idx]
        obs = self.seq_obs[path_idx, start:end]
        act = self.seq_act[path_idx, start:end]
        traj = np.concatenate([obs, act], -1)
        return {
            "x0": traj,
            "condition_cg": self.seq_val[path_idx, start] / self.scale,
        }


@hydra.main(config_path="../configs/diffuser", config_name="d4rl", version_base=None)
def pipeline(args):
    L.seed_everything(args.seed, workers=True)

    env_name = args.task.env_name
    save_path = Path(__file__).parents[1] / f"results/{args.pipeline_name}/{env_name}/"

    # --- Create Dataset ---
    env = gym.make(env_name)
    raw_dataset = env.get_dataset()
    if "kitchen" in env_name:
        dataset = D4RLKitchenDataset(raw_dataset, horizon=args.task.horizon, discount=0.99)
    elif "antmaze" in env_name:
        dataset = D4RLAntmazeDataset(raw_dataset, horizon=args.task.horizon, discount=0.99)
    else:
        dataset = D4RLMuJoCoDataset(raw_dataset, horizon=args.task.horizon, discount=0.99)
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    # --- Create Diffusion Model ---
    nn_diffusion = DiT1d(
        x_dim=obs_dim + act_dim, emb_dim=128, d_model=320, n_heads=10, depth=2, timestep_emb_type="untrainable_fourier"
    )
    nn_classifier = HalfJannerUNet1d(
        horizon=args.task.horizon,
        in_dim=obs_dim + act_dim,
        out_dim=1,
        dim_mult=args.task.dim_mult,
        timestep_emb_type="untrainable_fourier",
    )

    # --- Create Masks & Weights ---
    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.0
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = 10.0

    # --- Diffusion Training ---
    if args.mode == "training":
        classifier = OptimalityClassifier(nn_classifier, ema_rate=0.999)

        actor = ContinuousDiffusionSDE(nn_diffusion, None, fix_mask, loss_weight, ema_rate=0.999, classifier=classifier)

        dataloader = DataLoader(
            ObsActSequence_Wrapper(dataset, env_name),
            batch_size=512,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

        callback = ModelCheckpoint(
            dirpath=save_path, filename="diffusion-{step}", every_n_train_steps=args.save_interval, save_top_k=-1
        )

        trainer = L.Trainer(
            accelerator="gpu",
            devices=[0, 1, 2, 3],
            max_steps=args.diffusion_training_steps,
            deterministic=True,
            log_every_n_steps=200,
            default_root_dir=save_path,
            callbacks=[callback],
            strategy="ddp_find_unused_parameters_true",
        )

        trainer.fit(actor, dataloader)

    # --- Inference ---
    elif args.mode == "inference":
        num_envs = args.num_envs
        num_episodes = args.num_episodes
        num_candidates = args.num_candidates

        classifier = OptimalityClassifier(nn_classifier, ema_rate=0.999)

        actor = ContinuousDiffusionSDE(nn_diffusion, None, fix_mask, loss_weight, ema_rate=0.999, classifier=classifier)
        actor.load_state_dict(
            torch.load(save_path / f"diffusion-step={args.ckpt}.ckpt", map_location=f"cuda:{args.device_id}")["state_dict"]
        )
        actor.to(f"cuda:{args.device_id}").eval()

        env_eval = gym.vector.make(env_name, num_envs=num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((num_envs * num_candidates, args.task.horizon, obs_dim + act_dim))
        for i in range(num_episodes):
            obs, ep_reward, all_done, t = env_eval.reset(), 0, False, 0

            while not np.all(all_done) and t < 1000:
                obs = torch.tensor(normalizer.normalize(obs), dtype=torch.float32)
                obs = einops.repeat(obs, "b d -> (b k) d", k=num_candidates)
                prior[:, 0, :obs_dim] = obs

                traj, log = actor.sample(
                    prior,
                    solver=args.solver,
                    n_samples=num_envs * num_candidates,
                    sample_steps=args.sampling_steps,
                    condition_cg=None,
                    w_cg=args.task.w_cg,
                )

                logp = log["log_p"]
                traj = einops.rearrange(traj, "(b k) h d -> b k h d", k=num_candidates)
                logp = einops.rearrange(logp, "(b k) 1 -> b k 1", k=num_candidates).squeeze(-1)
                idx = torch.argmax(logp, dim=-1)
                traj = traj[torch.arange(num_envs), idx]
                act = traj[:, 0, obs_dim:].cpu().numpy()

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
