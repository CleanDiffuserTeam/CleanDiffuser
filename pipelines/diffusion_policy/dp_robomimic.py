from pathlib import Path

import gym
import h5py
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from cleandiffuser.dataset.robomimic_dataset import RobomimicDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.env import robomimic
from cleandiffuser.nn_condition import PearceObsCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import set_seed


class DatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        obs = item["lowdim"]
        act = item["action"]
        return {"x0": act, "condition_cfg": obs}


# --- Config ---
task = "transport"
quality = "mh"
abs_action = True
seed = 0
To = 2
Ta = 16
num_act_exec = 8
mode = "rendering"  # training, inference or rendering
dataset_path = (
    Path(__file__).parents[2]
    / f"dev/robomimic/datasets/{task}/{quality}/low_dim{'_abs' if abs_action else ''}.hdf5"
)
devices = [2]  # List of GPU ids to train on, cuda:0 for default
default_root_dir = (
    Path(__file__).parents[2] / f"results/diffusion_policy/robomimic_{task}_{quality}/"
)
training_steps = 500_000
save_every_n_steps = 50_000
ckpt_file = "step=500000.ckpt"
sampling_steps = 20

if __name__ == "__main__":
    set_seed(seed)

    # --- Dataset ---
    dataset = RobomimicDataset(dataset_dir=dataset_path, To=To, Ta=Ta, abs_action=abs_action)
    act_dim, lowdim_dim = dataset.action_dim, dataset.lowdim_dim

    dataloader = torch.utils.data.DataLoader(
        DatasetWrapper(dataset),
        batch_size=512,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    # --- Model ---
    nn_diffusion = DiT1d(
        x_dim=act_dim,
        emb_dim=128 * To,
        d_model=384,
        n_heads=6,
        depth=12,
        timestep_emb_type="untrainable_fourier",
    )
    nn_condition = PearceObsCondition(
        obs_dim=lowdim_dim,
        emb_dim=128,
        flatten=True,
        dropout=0.0,
    )

    policy = ContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion,
        nn_condition=nn_condition,
        x_max=torch.full((Ta, act_dim), 1.0),
        x_min=torch.full((Ta, act_dim), -1.0),
    )

    # -- Training ---
    if mode == "training":
        callback = ModelCheckpoint(
            dirpath=default_root_dir,
            every_n_train_steps=save_every_n_steps,
            save_top_k=-1,
        )
        trainer = L.Trainer(
            devices=devices,
            max_steps=training_steps,
            callbacks=[callback],
            default_root_dir=default_root_dir,
        )
        trainer.fit(policy, dataloader)

    # --- Inference ---
    elif mode == "inference":
        device = f"cuda:{devices[0]}"

        env = gym.make(
            "robomimic-v0",
            dataset_path=dataset_path,
            abs_action=abs_action,
            enable_render=False,
            use_image_obs=False,
        )
        normalizer = dataset.get_normalizer()

        policy.load_state_dict(torch.load(default_root_dir / ckpt_file)["state_dict"])
        policy = policy.to(device).eval()

        prior = torch.zeros((1, Ta, act_dim), device=device)

        success_rate = np.zeros(150)
        for k in range(150):
            obs, all_done, all_rew = env.reset(), False, 0
            obs = normalizer["lowdim"].normalize(obs["lowdim"][None,])
            obs = torch.tensor(obs, device=device, dtype=torch.float32)[:, None]
            obs = obs.repeat(1, To, 1)  # repeat padding for the first observation

            while not np.all(all_done):
                act, log = policy.sample(
                    prior, solver="ddpm", sample_steps=sampling_steps, condition_cfg=obs, w_cfg=1.0
                )
                act = normalizer["action"].unnormalize(act.cpu().numpy())
                act = dataset.action_converter.inverse_transform(act)

                for i in range(num_act_exec):
                    next_obs, rew, done, _ = env.step(act[0, i])
                    all_done = np.logical_or(all_done, done)
                    all_rew += rew

                    if i >= num_act_exec - To:
                        next_obs = normalizer["lowdim"].normalize(next_obs["lowdim"][None,])
                        next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
                        obs[:, i - num_act_exec + To] = next_obs

                    if np.all(all_done):
                        break

                print(f"[Test={k + 1}] Success:", all_rew)

            success_rate[k] += all_rew

        print(f"Success_rate: {success_rate.mean():.3f}")

    # -- Rendering ---
    elif mode == "rendering":
        import imageio

        device = f"cuda:{devices[0]}"

        env = gym.make(
            "robomimic-v0",
            dataset_path=dataset_path,
            abs_action=abs_action,
            enable_render=True,
            use_image_obs=False,
        )
        normalizer = dataset.get_normalizer()

        policy.load_state_dict(torch.load(default_root_dir / ckpt_file)["state_dict"])
        policy = policy.to(device).eval()

        prior = torch.zeros((1, Ta, act_dim), device=device)

        obs, all_done, all_rew = env.reset(), False, 0
        obs = normalizer["lowdim"].normalize(obs["lowdim"][None,])
        obs = torch.tensor(obs, device=device, dtype=torch.float32)[:, None]
        obs = obs.repeat(1, To, 1)  # repeat padding for the first observation

        frames = []
        while not np.all(all_done):
            act, log = policy.sample(
                prior, solver="ddpm", sample_steps=sampling_steps, condition_cfg=obs, w_cfg=1.0
            )
            act = normalizer["action"].unnormalize(act.cpu().numpy())
            act = dataset.action_converter.inverse_transform(act)

            for i in range(num_act_exec):
                next_obs, rew, done, _ = env.step(act[0, i])
                all_done = np.logical_or(all_done, done)
                all_rew += rew
                frames.append(env.render(mode="rgb_array"))

                if i >= num_act_exec - To:
                    next_obs = normalizer["lowdim"].normalize(next_obs["lowdim"][None,])
                    next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
                    obs[:, i - num_act_exec + To] = next_obs

                if np.all(all_done):
                    break

            print("Rewards:", all_rew)

        writer = imageio.get_writer(default_root_dir / "video.mp4", fps=30)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        env.close()
