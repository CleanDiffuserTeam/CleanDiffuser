from pathlib import Path

import d4rl
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
    def __init__(self, dataset: torch.utils.data.Dataset, env_name: str, need_value: bool = False):
        self.need_value = need_value
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
        if self.need_value:
            return {
                "x0": traj,
                "condition_cg": self.seq_val[path_idx, start] / self.scale, }
        else:
            return {"x0": traj}


@hydra.main(config_path="../configs/diffuser", config_name="d4rl", version_base=None)
def pipeline(args):

    L.seed_everything(args.seed, workers=True)

    env_name = args.task.env_name
    save_path = Path(__file__).parents[1] / \
        f"results/{args.pipeline_name}/{env_name}/"

    # --- Create Dataset ---
    env = gym.make(env_name)
    raw_dataset = env.get_dataset()
    if "kitchen" in env_name:
        dataset = D4RLKitchenDataset(
            raw_dataset, horizon=args.task.horizon, discount=0.99)
    elif "antmaze" in env_name:
        dataset = D4RLAntmazeDataset(
            raw_dataset, horizon=args.task.horizon, discount=0.99)
    else:
        dataset = D4RLMuJoCoDataset(
            raw_dataset, horizon=args.task.horizon, discount=0.99)
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    # --- Create Diffusion Model ---
    nn_diffusion = DiT1d(
        in_dim=obs_dim + act_dim, emb_dim=128, d_model=320, n_heads=10, depth=2,
        timestep_emb_type="untrainable_fourier")
    nn_classifier = HalfJannerUNet1d(
        horizon=args.task.horizon, in_dim=obs_dim + act_dim, out_dim=1,
        dim_mult=args.task.dim_mult, timestep_emb_type="untrainable_fourier")

    # --- Create Masks & Weights ---
    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = 10.

    # --- Diffusion Training ---
    if args.mode == "diffusion_training":

        classifier = OptimalityClassifier(nn_classifier, ema_rate=0.9999)

        actor = ContinuousDiffusionSDE(
            nn_diffusion, None, fix_mask, loss_weight,
            ema_rate=0.9999, classifier=classifier)

        dataloader = DataLoader(
            ObsActSequence_Wrapper(dataset, env_name, need_value=True),
            batch_size=64, shuffle=True, num_workers=4, persistent_workers=True)

        callback = ModelCheckpoint(
            dirpath=save_path, filename="diffusion-{step}",
            every_n_train_steps=args.save_interval)

        trainer = L.Trainer(
            accelerator='gpu', devices=[args.device_id,],
            max_steps=args.diffusion_training_steps, deterministic=True, log_every_n_steps=1000,
            default_root_dir=save_path, callbacks=[callback])

        trainer.fit(actor, dataloader)


if __name__ == "__main__":
    pipeline()
