from pathlib import Path

import d4rl
import gym
import hydra
import pytorch_lightning as L
import torch
import torch.utils
import torch.utils.data
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeTDDataset
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenTDDataset
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import IDQLMlp


class Transition_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        obs = self.obs[idx]
        next_obs = self.next_obs[idx]
        act = self.act[idx]
        rew = self.rew[idx]
        tml = self.tml[idx]
        return {"x0": torch.cat([obs, next_obs, rew, act, tml], -1)}


@hydra.main(config_path="../configs/synther", config_name="d4rl", version_base=None)
def pipeline(args):

    L.seed_everything(args.seed, workers=True)

    env_name = args.task.env_name
    save_path = Path(__file__).parents[1] / \
        f"results/{args.pipeline_name}/{env_name}/"

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

    """
    SynthER generates transitions, which are aranged as
    (obs, next_obs, rew, act, tml).
    So the dimension is `(obs_dim + obs_dim + 1 + act_dim + 1)`,
    such that the last `act_dim + 1` dimensions are ranged in `[-1, 1]`.
    """
    x_dim = 2 * obs_dim + act_dim + 2

    # --- Create Loss-weights ---
    """
    Although `obs` can have a larger dimension than `rew` and `act`,
    they are equally important in the training.
    So we set the loss-weight according to the dimension of the data.
    """
    loss_weight = torch.empty((x_dim,))
    loss_weight[:2 * obs_dim] = x_dim / obs_dim
    loss_weight[2 * obs_dim:2 * obs_dim + 1] = x_dim / 1
    loss_weight[2 * obs_dim + 1:2 * obs_dim + 1 + act_dim] = x_dim / act_dim
    loss_weight[-1] = x_dim
    loss_weight = loss_weight / loss_weight.mean()
    x_max = torch.empty((x_dim,))
    x_min = torch.empty((x_dim,))
    x_max[:2 * obs_dim + 1] = torch.inf
    x_max[2 * obs_dim + 1:] = 1
    x_min[:2 * obs_dim + 1] = -torch.inf
    x_min[2 * obs_dim + 1:] = -1
    x_min[-1] = 0

    # --- Create Diffusion Model ---
    nn_diffusion = IDQLMlp(
        x_dim=x_dim, emb_dim=128, hidden_dim=1024, n_blocks=6, dropout=0.1,
        timestep_emb_type="untrainable_fourier")

    # --- SynthER Training ---
    if args.mode == "synther_training":

        synther = ContinuousDiffusionSDE(
            nn_diffusion, loss_weight=loss_weight, ema_rate=0.9999,
            x_max=x_max, x_min=x_min)

        dataloader = DataLoader(
            Transition_Wrapper(dataset),
            batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)

        callback = ModelCheckpoint(
            dirpath=save_path, filename="synther-{step}",
            every_n_train_steps=args.save_interval)

        trainer = L.Trainer(
            accelerator='gpu', devices=[args.device_id,],
            max_steps=args.diffusion_training_steps, deterministic=True, log_every_n_steps=1000,
            default_root_dir=save_path, callbacks=[callback])

        trainer.fit(synther, dataloader)

    # --- Dataset Upsampling ---
    elif args.mode == "dataset_upsampling":

        synther = ContinuousDiffusionSDE.load_from_checkpoint(
            checkpoint_path=save_path / "synther-step=1000000.ckpt",
            nn_diffusion=nn_diffusion, loss_weight=loss_weight, ema_rate=0.9999,
            x_max=x_max, x_min=x_min)

        pass


if __name__ == "__main__":
    pipeline()
