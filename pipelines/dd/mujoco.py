from typing import Any

import d4rl
import gym
import pytorch_lightning as L
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
import numpy as np
import hydra

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset
from cleandiffuser.diffusion.diffusionsde_L import ContinuousDiffusionSDE
from cleandiffuser.invdynamic.mlp_L import FancyMlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE
from pytorch_lightning.profilers import AdvancedProfiler
from pathlib import Path

DD_RETURN_SCALE = {
    "halfcheetah-medium-expert-v2": 3600,
    "halfcheetah-medium-replay-v2": 1600,
    "halfcheetah-medium-v2": 1700,
    "hopper-medium-expert-v2": 1200,
    "hopper-medium-replay-v2": 1000,
    "hopper-medium-v2": 1000,
    "walker2d-medium-expert-v2": 1600,
    "walker2d-medium-replay-v2": 1300,
    "walker2d-medium-v2": 1300,
    "kitchen-partial-v0": 470,
    "kitchen-mixed-v0": 400,
    "antmaze-medium-play-v2": 100,
    "antmaze-medium-diverse-v2": 100,
    "antmaze-large-play-v2": 100,
    "antmaze-large-diverse-v2": 100,
}


class DDD4RLMuJoCoDataset(D4RLMuJoCoDataset):
    return_scale = 1.

    def set_return_scale(self, return_scale):
        self.return_scale = return_scale

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]
        seq_obs = self.seq_obs[path_idx, start:end]
        return seq_obs, self.seq_val[path_idx, start] / self.return_scale


@hydra.main(config_path="../../configs_L/dd/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):
    L.seed_everything(args.seed, workers=True)

    env_name = args.task.env_name
    horizon = args.task.horizon
    save_path = Path(__file__).parents[2] / f"results/{args.pipeline_name}/mujoco/{env_name}/"
    mode = args.mode

    # ------------------- Create Dataset -------------------------
    env = gym.make(env_name)
    dataset = DDD4RLMuJoCoDataset(env.get_dataset(), horizon=horizon, discount=0.997, terminal_penalty=-100)
    dataset.set_return_scale(DD_RETURN_SCALE[env_name])
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = DiT1d(
        obs_dim + act_dim, emb_dim=128,
        d_model=320, n_heads=10, depth=2, timestep_emb_type="untrainable_fourier")
    nn_condition = MLPCondition(
        in_dim=1, out_dim=128, hidden_dims=[128, ], act=nn.SiLU(), dropout=0.2)

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = 10.

    # ------------------- Training --------------------
    if mode == "training":
        agent = ContinuousDiffusionSDE(
            nn_diffusion, nn_condition, fix_mask, loss_weight, ema_rate=0.9999,
            optim_params={"lr": 3e-4})

        trainer = L.Trainer(
            accelerator='gpu', devices="auto", max_steps=10000, deterministic=True, log_every_n_steps=1000,
            profiler="simple", strategy="ddp", default_root_dir=save_path)
        trainer.fit(agent, dataloader)

    elif mode == "inference":

        target_return = 0.95
        w_cfg = 1.0

        agent = ContinuousDiffusionSDE.load_from_checkpoint(
            checkpoint_path="lightning_logs/version_10/checkpoints/2.ckpt",
            map_location=device,
            nn_diffusion=nn_diffusion, nn_condition=nn_condition,
            fix_mask=fix_mask, loss_weight=loss_weight, ema_rate=0.9999)
        agent.to(device).eval()

        env_eval = gym.vector.make(env_name, num_envs=50)
        normalizer = dataset.get_normalizer()

        prior = torch.zeros((50, horizon, obs_dim + act_dim))
        condition = torch.ones((50, 1), device=device) * target_return

        obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
        while not np.all(cum_done) and t < 1000 + 1:
            # normalize obs
            obs = torch.tensor(normalizer.normalize(obs), device=device, dtype=torch.float32)

            # sample trajectories
            prior[:, 0, :obs_dim] = obs
            traj, log = agent.sample(
                prior, solver="ddim",
                n_samples=50, sample_steps=20, use_ema=True, sample_step_schedule="quad_continuous",
                condition_cfg=condition, w_cfg=w_cfg, temperature=0.5)

            # inverse dynamic
            act = traj[:, 0, obs_dim:].clamp(-1., 1.).cpu().numpy()

            # step
            obs, rew, done, info = env_eval.step(act)

            t += 1
            cum_done = done if cum_done is None else np.logical_or(cum_done, done)
            ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew
            print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

        env_eval.close()


if __name__ == "__main__":
    pipeline()
