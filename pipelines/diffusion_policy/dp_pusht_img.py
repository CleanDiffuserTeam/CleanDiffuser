from pathlib import Path

import gym
import numpy as np
import pytorch_lightning as L
import torch
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint

from cleandiffuser.dataset.pusht_dataset import PushTImageDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.env import pusht
from cleandiffuser.nn_condition import ResNet18MultiViewImageCondition
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import set_seed


class DatasetWrapper:
    def __init__(self, dataset, To, Ta):
        self.dataset = dataset
        self.To, self.Ta = To, Ta
        self.random_crop = T.RandomCrop((84, 84))

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        obs = self.random_crop(item["obs"]["image"][: self.To]).unsqueeze(0)
        act = item["action"][-self.Ta :]
        return {"x0": act, "condition_cfg": obs}


# ----- Config -----
seed = 0
To = 2  # observation horizon
Ta = 16  # action horizon
num_act_exec = 8
mode = "inference"  # training, inference or rendering
dataset_path = Path(__file__).parents[2] / "dev/pusht/pusht_cchi_v7_replay.zarr"
devices = [3]  # List of GPU ids to train on, cuda:0 for default
default_root_dir = Path(__file__).parents[2] / "results/diffusion_policy/pusht_img/"
training_steps = 100_000
save_every_n_steps = 10_000
ckpt_file = "step=100000.ckpt"
sampling_steps = 20


if __name__ == "__main__":
    set_seed(seed)

    # --- Dataset ---
    dataset = PushTImageDataset(
        zarr_path=dataset_path,
        horizon=To + Ta - 1,
        pad_before=To - 1,
        pad_after=Ta - 1,
    )
    act_dim = dataset.act_dim

    dataloader = torch.utils.data.DataLoader(
        DatasetWrapper(dataset, To, Ta),
        batch_size=512,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    # --- Model ---
    nn_diffusion = DiT1d(
        x_dim=act_dim,
        emb_dim=128 * To,
        d_model=256,
        n_heads=4,
        depth=4,
        timestep_emb_type="untrainable_fourier",
    )
    nn_condition = ResNet18MultiViewImageCondition(
        image_sz=84,
        in_channel=3,
        emb_dim=128,
        n_views=1,
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

    # -- Inference ---
    elif mode == "inference":
        device = f"cuda:{devices[0]}"
        env = gym.vector.make("pusht-image-v0", num_envs=50)
        normalizer = dataset.get_normalizer()
        center_crop = T.CenterCrop((84, 84))

        policy.load_state_dict(torch.load(default_root_dir / ckpt_file)["state_dict"])
        policy = policy.to(device).eval()

        prior = torch.zeros((50, Ta, act_dim), device=device)

        success_rate = np.zeros(5)
        for k in range(5):
            obs, all_done, success, all_rew = env.reset(), False, False, 0
            obs = normalizer["obs"]["image"].normalize(obs["image"])
            obs = torch.tensor(obs, device=device, dtype=torch.float32)[:, None]
            obs = center_crop(obs)
            obs = obs.repeat(1, To, 1, 1, 1)  # repeat padding for the first observation

            while not np.all(all_done):
                act, log = policy.sample(
                    prior,
                    solver="ddpm",
                    sample_steps=sampling_steps,
                    condition_cfg=obs[:, None],
                    w_cfg=1.0,
                    sampling_schedule="quad",
                )
                act = normalizer["action"].unnormalize(act.cpu().numpy())

                for i in range(num_act_exec):
                    next_obs, rew, done, _ = env.step(act[:, i])
                    all_done = np.logical_or(all_done, done)
                    success = np.logical_or(success, rew > 0.95)
                    all_rew += rew

                    if i >= num_act_exec - To:
                        next_obs = normalizer["obs"]["image"].normalize(next_obs["image"])
                        next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
                        next_obs = center_crop(next_obs)
                        obs[:, i - num_act_exec + To] = next_obs

                    if np.all(all_done):
                        break

                print("Success:", success)

            success_rate[k] += success.mean()

        print(f"Success_rate: {success_rate.mean():.3f}±{success_rate.std():.3f}")

        env.close()

    # --- Rendering ---
    elif mode == "rendering":
        import imageio

        device = f"cuda:{devices[0]}"

        env = gym.make("pusht-image-v0")
        normalizer = dataset.get_normalizer()
        center_crop = T.CenterCrop((84, 84))

        policy.load_state_dict(torch.load(default_root_dir / ckpt_file)["state_dict"])
        policy = policy.to(device).eval()

        prior = torch.zeros((1, Ta, act_dim), device=device)

        obs, all_done, success, all_rew = env.reset(), False, False, 0
        obs = normalizer["obs"]["image"].normalize(obs["image"][None,])
        obs = torch.tensor(obs, device=device, dtype=torch.float32)[:, None]
        obs = center_crop(obs)
        obs = obs.repeat(1, To, 1, 1, 1)  # repeat padding for the first observation

        frames = []
        while not np.all(all_done):
            act, log = policy.sample(
                prior,
                solver="ddpm",
                sample_steps=sampling_steps,
                condition_cfg=obs[:, None],
                w_cfg=1.0,
                sampling_schedule="quad",
            )
            act = normalizer["action"].unnormalize(act.cpu().numpy())

            for i in range(num_act_exec):
                next_obs, rew, done, _ = env.step(act[0, i])
                all_done = np.logical_or(all_done, done)
                success = np.logical_or(success, rew > 0.95)
                all_rew += rew
                frames.append(env.render(mode="rgb_array"))

                if i >= num_act_exec - To:
                    next_obs = normalizer["obs"]["image"].normalize(next_obs["image"][None,])
                    next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
                    next_obs = center_crop(next_obs)
                    obs[:, i - num_act_exec + To] = next_obs

                if np.all(all_done):
                    break

            print("Success:", success)

        writer = imageio.get_writer(default_root_dir / "video.mp4", fps=30)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        env.close()
