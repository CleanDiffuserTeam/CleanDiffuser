from pathlib import Path

import d4rl
import gym
import hydra
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import IQL


class BC_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        return self.act[idx], self.obs[idx]


class IQL_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.tml[idx]


@hydra.main(config_path="../configs/idql", config_name="d4rl", version_base=None)
def pipeline(args):

    L.seed_everything(args.seed, workers=True)

    env_name = args.task.env_name
    save_path = Path(__file__).parents[1] / f"results/{args.pipeline_name}/{env_name}/"

    # --- Create Dataset ---
    env = gym.make(env_name)
    raw_dataset = d4rl.qlearning_dataset(env)
    dataset = D4RLMuJoCoTDDataset(raw_dataset, args.normalize_reward)
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    # --- Create Diffusion Model ---
    nn_diffusion = IDQLMlp(
        x_dim=act_dim, emb_dim=64, hidden_dim=256, n_blocks=3, dropout=0.1,
        timestep_emb_type="untrainable_fourier")
    nn_condition = MLPCondition(
        in_dim=obs_dim, out_dim=64, hidden_dims=[64, ], act=torch.nn.SiLU(), dropout=0.0)

    # --- Training ---
    if args.mode == "bc_training":

        actor = ContinuousDiffusionSDE(
            nn_diffusion, nn_condition, ema_rate=0.9999,
            x_max=+1.0*torch.ones((act_dim, )),
            x_min=-1.0*torch.ones((act_dim, )))

        dataloader = DataLoader(
            BC_Wrapper(dataset),
            batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)

        callback = ModelCheckpoint(
            dirpath=save_path, filename="diffusion_bc-{step}",
            every_n_train_steps=args.save_interval)

        trainer = L.Trainer(
            accelerator='gpu', devices=[args.device_id,],
            max_steps=args.bc_training_steps, deterministic=True, log_every_n_steps=1000,
            default_root_dir=save_path, callbacks=[callback])

        trainer.fit(actor, dataloader)

    elif args.mode == "iql_training":

        iql = IQL(obs_dim, act_dim, tau=args.task.iql_tau,
                  discount=0.99, hidden_dim=256)

        dataloader = DataLoader(
            IQL_Wrapper(dataset),
            batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)

        callback = ModelCheckpoint(
            dirpath=save_path, filename="iql-{step}",
            every_n_train_steps=args.save_interval)

        trainer = L.Trainer(
            accelerator='gpu', devices=[args.device_id,],
            max_steps=args.iql_training_steps, deterministic=True, log_every_n_steps=1000,
            default_root_dir=save_path, callbacks=[callback])

        trainer.fit(iql, dataloader)

    elif args.mode == "inference":

        actor = ContinuousDiffusionSDE.load_from_checkpoint(
            checkpoint_path=save_path /
            "lightning_logs/version_1/checkpoints/epoch=0-step=7805.ckpt",
            nn_diffusion=nn_diffusion, nn_condition=nn_condition, ema_rate=0.9999,
            x_max=+1.0*torch.ones((act_dim, )),
            x_min=-1.0*torch.ones((act_dim, )))


if __name__ == "__main__":
    pipeline()
