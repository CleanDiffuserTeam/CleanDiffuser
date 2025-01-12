from pathlib import Path
from typing import Dict, Optional

import gym
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint

from cleandiffuser.dataset.robomimic_dataset import RobomimicDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.env import robomimic
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_condition.resnets import ResNet18
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import set_seed


class MultiViewResnetWithLowdimObsCondition(IdentityCondition):
    """MultiViewResnetWithLowdimObsCondition.

    Since Robomimic Image Benchmark provides multi-view images and low-dim eef states,
    we use a ResNet18 to extract image features and a MLP to extract low-dim features.

    Args:
        image_sz (int):
            Image size.
        in_channel (int):
            Number of input channels.
        lowdim (int):
            Dimension of low-dim observation.
        image_emb_dim (int):
            Dimension of image embedding.
        lowdim_emb_dim (int):
            Dimension of low-dim embedding.
        dropout (float):
            Classifier-free guidance condition dropout rate.

    Examples:
        >>> nn_condition = MultiViewResnetWithLowdimObsCondition()
        >>> condition = {
            "image": torch.randn((5, 2, 2, 3, 64, 64)),  # (b, To, n_views, C, H, W)
            "lowdim": torch.randn((5, 2, 7)),  # (b, To, lowdim)
        }
        >>> nn_condition(condition).shape
        torch.Size([5, 1152])  # (2 * 256 + 64) * 2
    """

    def __init__(
        self,
        image_sz: int = 76,
        in_channel: int = 3,
        lowdim: int = 9,
        image_emb_dim: int = 256,
        lowdim_emb_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__(dropout)
        self.resnet18 = ResNet18(image_sz=image_sz, in_channel=in_channel, emb_dim=image_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(lowdim, lowdim_emb_dim), nn.SiLU(), nn.Linear(lowdim_emb_dim, lowdim_emb_dim)
        )

    def forward(self, condition: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None):
        image = condition["image"]
        lowdim = condition["lowdim"]

        leading_dims = image.shape[:-3]
        image = image.reshape(-1, *image.shape[-3:])
        image_feat = self.resnet18(image)
        image_feat = image_feat.reshape(leading_dims + image_feat.shape[-1:])
        image_feat = torch.flatten(image_feat, start_dim=1)

        lowdim_feat = torch.flatten(self.mlp(lowdim), start_dim=1)

        cond_feat = torch.cat([image_feat, lowdim_feat], dim=-1)

        return super().forward(cond_feat, mask)


class DatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
        self.random_crop = T.RandomCrop(76)

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        image = (image / 255.0).astype(np.float32)
        image = self.random_crop(torch.tensor(image))

        lowdim = item["lowdim"]
        act = item["action"]
        return {
            "x0": act,
            "condition_cfg": {
                "image": image,
                "lowdim": lowdim,
            },
        }


# --- Config ---
task = "can"
quality = "ph"
abs_action = True
seed = 0
To = 2
Ta = 16
num_act_exec = 8
mode = "rendering"  # training, inference or rendering
dataset_path = (
    Path(__file__).parents[2]
    / f"dev/robomimic/datasets/{task}/{quality}/image{'_abs' if abs_action else ''}.hdf5"
)
devices = [1]  # List of GPU ids to train on, cuda:0 for default
default_root_dir = (
    Path(__file__).parents[2] / f"results/diffusion_policy/robomimic_img_{task}_{quality}/"
)
training_steps = 500_000  # for hard tasks like transport, 500k steps may not be enough
save_every_n_steps = 50_000
ckpt_file = "step=500000.ckpt"
sampling_steps = 20


if __name__ == "__main__":
    set_seed(seed)

    # --- Dataset ---
    dataset = RobomimicDataset(dataset_dir=dataset_path, To=To, Ta=Ta, abs_action=abs_action)
    act_dim, lowdim_dim = dataset.action_dim, dataset.lowdim_dim
    n_views = 3 if task == "transport" else 2

    dataloader = torch.utils.data.DataLoader(
        DatasetWrapper(dataset),
        batch_size=64,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    # --- Model ---
    nn_diffusion = DiT1d(
        x_dim=act_dim,
        emb_dim=To * (n_views * 256 + 64),
        d_model=384,
        n_heads=6,
        depth=12,
        timestep_emb_type="untrainable_fourier",
    )
    nn_condition = MultiViewResnetWithLowdimObsCondition(
        image_sz=76, in_channel=3, lowdim=lowdim_dim, image_emb_dim=256, lowdim_emb_dim=64
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

    # -- rendering --
    elif mode == "rendering":
        import imageio

        device = f"cuda:{devices[0]}"

        env = gym.make(
            "robomimic-v0",
            dataset_path=dataset_path,
            abs_action=abs_action,
            enable_render=True,
            use_image_obs=True,
        )
        normalizer = dataset.get_normalizer()
        center_crop = T.CenterCrop(76)

        policy.load_state_dict(torch.load(default_root_dir / ckpt_file)["state_dict"])
        policy = policy.to(device).eval()

        prior = torch.zeros((1, Ta, act_dim), device=device)
        obs, all_done, all_rew = env.reset(), False, 0

        lowdim = normalizer["lowdim"].normalize(obs["lowdim"][None,])
        lowdim = torch.tensor(lowdim, device=device, dtype=torch.float32)[:, None]
        lowdim = lowdim.repeat(1, To, 1)  # repeat padding for the first observation

        image = torch.tensor(obs["image"] / 255.0, device=device, dtype=torch.float32)[
            None, None, :
        ]
        image = center_crop(image)
        image = image.repeat(1, To, 1, 1, 1, 1)

        frames = []
        while not np.all(all_done):
            act, log = policy.sample(
                prior,
                solver="ddpm",
                sample_steps=sampling_steps,
                condition_cfg={"image": image, "lowdim": lowdim},
                w_cfg=1.0,
            )
            act = normalizer["action"].unnormalize(act.cpu().numpy())
            act = dataset.action_converter.inverse_transform(act)

            for i in range(num_act_exec):
                next_obs, rew, done, _ = env.step(act[0, i])
                all_done = np.logical_or(all_done, done)
                all_rew += rew
                frames.append(env.render(mode="rgb_array"))

                if i >= num_act_exec - To:
                    this_lowdim = normalizer["lowdim"].normalize(next_obs["lowdim"][None,])
                    this_lowdim = torch.tensor(this_lowdim, device=device, dtype=torch.float32)
                    lowdim[:, i - num_act_exec + To] = this_lowdim

                    this_image = torch.tensor(
                        next_obs["image"] / 255.0, device=device, dtype=torch.float32
                    )[None,]
                    this_image = center_crop(this_image)
                    image[:, i - num_act_exec + To] = this_image

                if np.all(all_done):
                    break

            print("Rewards:", all_rew)

        writer = imageio.get_writer(default_root_dir / "video.mp4", fps=30)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        env.close()
