from pathlib import Path

import gym
import gym.vector
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from cleandiffuser.dataset.kitchen_dataset import KitchenDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.env import kitchen
from cleandiffuser.nn_condition import PearceObsCondition
from cleandiffuser.nn_diffusion import PearceMlp
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
        obs = item["state"]
        act = item["action"][0]
        return {"x0": act, "condition_cfg": obs}


# ----- Config -----
seed = 0
mode = "rendering"  # training, inference or rendering
dataset_path = Path(__file__).parents[2] / "dev/kitchen"
abs_action = False  # True for position control, False for velocity control
devices = [0]  # List of GPU ids to train on, cuda:0 for default
default_root_dir = Path(__file__).parents[2] / "results/dbc/kitchen/"
training_steps = 200_000
save_every_n_steps = 40_000
ckpt_file = "epoch=793-step=200000.ckpt"
sampling_steps = 5

if __name__ == "__main__":
    set_seed(seed)

    # --- Dataset ---
    # Take one history observation
    dataset = KitchenDataset(
        dataset_dir=dataset_path,
        horizon=1,
        abs_action=abs_action,
    )
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    dataloader = torch.utils.data.DataLoader(
        DatasetWrapper(dataset),
        batch_size=512,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    # --- Model ---
    nn_diffusion = PearceMlp(
        x_dim=act_dim, emb_dim=128, condition_horizon=1, timestep_emb_type="untrainable_fourier"
    )
    nn_condition = PearceObsCondition(obs_dim=obs_dim, emb_dim=128, flatten=False, dropout=0.0)

    policy = ContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion,
        nn_condition=nn_condition,
        x_max=torch.full((act_dim,), 1.0),
        x_min=torch.full((act_dim,), -1.0),
    )

    # --- Training ---
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

        env = gym.vector.make("kitchen-all-v0", num_envs=50, use_abs_action=abs_action)
        normalizer = dataset.get_normalizer()

        policy.load_state_dict(torch.load(default_root_dir / ckpt_file)["state_dict"])
        policy = policy.to(device).eval()

        prior = torch.zeros((50, act_dim), device=device)

        success_rate = np.zeros((5, 5))
        for k in range(5):
            obs, all_done, all_rew = env.reset(), False, 0
            obs = normalizer["state"].normalize(obs)
            obs = torch.tensor(obs, device=device, dtype=torch.float32)[:, None]

            while not np.all(all_done):
                act, _ = policy.sample(
                    prior=prior,
                    solver="ddpm",
                    sample_steps=sampling_steps,
                    condition_cfg=obs,
                    w_cfg=1.0,
                )
                act = normalizer["action"].unnormalize(act.cpu().numpy())

                next_obs, rew, done, _ = env.step(act)
                all_done = np.logical_or(all_done, done)
                all_rew += rew

                next_obs = normalizer["state"].normalize(next_obs)
                next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
                obs[:, 0] = next_obs

                print("Rewards:", all_rew)

            for i in range(5):
                success_rate[k, i] += (all_rew > i).sum() / 50

        print(
            "Success_rate:",
            [f"{mean}±{std}" for mean, std in zip(success_rate.mean(0), success_rate.std(0))],
        )

        env.close()

    # --- Rendering ---
    elif mode == "rendering":
        import imageio

        device = f"cuda:{devices[0]}"

        env = gym.make("kitchen-all-v0", use_abs_action=abs_action)
        normalizer = dataset.get_normalizer()

        policy.load_state_dict(torch.load(default_root_dir / ckpt_file)["state_dict"])
        policy = policy.to(device).eval()

        obs, all_done, all_rew = env.reset(), False, 0
        obs = normalizer["state"].normalize(obs[None,])
        obs = torch.tensor(obs, device=device, dtype=torch.float32)[:, None]

        prior = torch.zeros((1, act_dim), device=device)
        frames = []

        while not all_done:
            act, _ = policy.sample(
                prior=prior,
                solver="ddpm",
                sample_steps=sampling_steps,
                condition_cfg=obs,
                w_cfg=1.0,
            )
            act = normalizer["action"].unnormalize(act.cpu().numpy())

            next_obs, rew, done, _ = env.step(act[0])
            all_done = all_done or done
            all_rew += rew
            frames.append(env.render("rgb_array"))

            next_obs = normalizer["state"].normalize(next_obs[None,])
            next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
            obs[:, 0] = next_obs

            print(all_rew)

        writer = imageio.get_writer(default_root_dir / "video.mp4", fps=30)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        env.close()
