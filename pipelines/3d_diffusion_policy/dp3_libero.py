from pathlib import Path
from typing import Dict, List, Optional

import einops
import gym
import h5py
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint
from timm.models import VisionTransformer
from transformers import T5Config, T5EncoderModel, T5Tokenizer

from cleandiffuser.dataset.libero_dataset import LiberoDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.env import libero
from cleandiffuser.nn_condition import BaseNNCondition, DP3PointCloudCondition
from cleandiffuser.nn_diffusion import DiT1dWithCrossAttention
from cleandiffuser.utils import UntrainablePositionalEmbedding, set_seed


class PointNetAndT5PCLanguageCondition(DP3PointCloudCondition):
    def __init__(
        self,
        fps_downsample_points: Optional[int] = 1024,
        pointnet_hidden_sizes: List[int] = (64, 128, 256),
        t5_pretrained_model_name_or_path: str = "google-t5/t5-base",
        t5_hidden_dim: int = 768,
        t5_max_seq_len: int = 32,
        emb_dim: int = 384,
        To: int = 2,
    ):
        super().__init__(
            emb_dim=emb_dim,
            fps_downsample_points=fps_downsample_points,
            hidden_sizes=pointnet_hidden_sizes,
        )
        self.t5_adapter = nn.Sequential(
            nn.Linear(t5_hidden_dim, emb_dim), nn.GELU(approximate="tanh")
        )
        t5_pos_emb = (
            UntrainablePositionalEmbedding(emb_dim, 100)(torch.arange(t5_max_seq_len))[None] * 0.2
        )
        self.t5_pos_emb = nn.Parameter(t5_pos_emb)

        self.To_pos_emb = nn.Parameter(torch.randn(1, To, emb_dim) * 0.02)
        self.pc_adapter = nn.Linear(emb_dim * To, emb_dim)

    def forward(self, condition: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None):
        pointcloud = condition["pointcloud"]
        b = pointcloud.shape[0]

        language_embedding = condition["language_embedding"]
        language_mask = condition["language_mask"]

        # transform to pytorch attention padding mask, where True means padding.
        if language_mask is not None:
            language_mask = torch.logical_not(language_mask.to(torch.bool))

        pc_feat = super().forward(einops.rearrange(pointcloud, "b t n d -> (b t) n d"))
        pc_feat = einops.rearrange(pc_feat, "(b t) d -> b t d", b=b)
        pc_feat = pc_feat + self.To_pos_emb
        pc_feat = einops.rearrange(pc_feat, "b t d -> b (t d)")
        pc_feat = self.pc_adapter(pc_feat)

        lang_feat = self.t5_adapter(language_embedding) + self.t5_pos_emb

        return {
            "vec_condition": pc_feat,
            "seq_condition": lang_feat,
            "seq_condition_mask": language_mask,
        }


class DatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        pointcloud = item["observation"]["pointcloud"].astype(np.float32)
        act = item["action"]
        obs = {
            "pointcloud": pointcloud,
            "language_embedding": item["language_embedding"],
            "language_mask": item["language_mask"],
        }
        return {"x0": act, "condition_cfg": obs}


# --- Config ---
task_suite = "libero_goal"
t5_pretrained_model_name_or_path = "/home/dzb/pretrained/t5-base"
seed = 0
To = 1
Ta = 16
num_act_exec = 8
mode = "training"  # training, inference or rendering
dataset_path = Path(__file__).parents[2] / f"dev/libero/{task_suite}.zarr"
devices = [0, 1, 2, 3]  # List of GPU ids to train on, cuda:0 for default
default_root_dir = Path(__file__).parents[2] / f"results/3d_diffusion_policy/{task_suite}/"
training_steps = 50_000
save_every_n_steps = 5_000
ckpt_file = "epoch=331-step=165000.ckpt"
env_name = "libero-goal-v0"
task_id = 1
sampling_steps = 20

if __name__ == "__main__":
    set_seed(seed)

    # --- Dataset ---
    dataset = LiberoDataset(
        data_path=dataset_path,
        observation_meta=["pointcloud"],
        To=To,
        Ta=Ta,
    )
    act_dim = 7

    dataloader = torch.utils.data.DataLoader(
        DatasetWrapper(dataset),
        batch_size=64,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )

    # --- Model ---
    t5_hidden_dim = T5Config.from_pretrained(t5_pretrained_model_name_or_path).d_model

    nn_diffusion = DiT1dWithCrossAttention(
        x_dim=act_dim,
        x_seq_len=Ta,
        emb_dim=384,
        d_model=384,
        n_heads=6,
        depth=12,
        timestep_emb_type="untrainable_fourier",
        timestep_emb_params={"scale": 0.2},
    )
    nn_condition = PointNetAndT5PCLanguageCondition(
        fps_downsample_points=1024,
        t5_pretrained_model_name_or_path=t5_pretrained_model_name_or_path,
        t5_hidden_dim=t5_hidden_dim,
        t5_max_seq_len=32,
        emb_dim=384,
        To=To,
    )

    policy = ContinuousDiffusionSDE(
        nn_diffusion=nn_diffusion,
        nn_condition=nn_condition,
        ema_rate=0.75,
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
            precision="bf16-mixed",
        )
        trainer.fit(policy, dataloader)

    # -- rendering --
    elif mode == "rendering":
        import imageio

        device = f"cuda:{devices[0]}"
        lang_encoder = T5LanguageEncoder(
            pretrained_model_name_or_path=t5_pretrained_model_name_or_path, device=device
        )

        env = gym.make(
            env_name,
            task_id=task_id,
            image_size=224,
            require_depth=True,
            require_point_cloud=True,
            seed=seed,
            enable_pytorch3d_fps=True,
            pointcloud_process_device=device,
        )
        print(env.task_description)
        normalizer = dataset.get_normalizer()

        policy.load_state_dict(torch.load(default_root_dir / ckpt_file)["state_dict"])
        policy = policy.to(device).eval()

        prior = torch.zeros((1, Ta, act_dim), device=device)
        obs, all_done, all_rew = env.reset(), False, 0

        lang_emb, lang_mask = lang_encoder.encode([env.task_description])
        lang_emb = lang_emb.cpu().numpy()[0]
        lang_mask = lang_mask.cpu().numpy()[0]
        if lang_emb.shape[0] < 32:
            lang_emb = np.pad(lang_emb, ((0, 32 - lang_emb.shape[0]), (0, 0)))
            lang_mask = np.pad(lang_mask, (0, 32 - lang_mask.shape[0]))
        elif lang_emb.shape[0] > 32:
            lang_emb = lang_emb[:32, :]
            lang_mask = lang_mask[:32]
        lang_emb = torch.tensor(lang_emb, device=device)[None]
        lang_mask = torch.tensor(lang_mask, device=device)[None]

        pointcloud = torch.tensor(obs["agentview_pointcloud"], device=device, dtype=torch.float32)
        pointcloud = pointcloud[None, None].repeat(1, To, 1, 1)

        frames = []
        frames.append(
            np.concatenate([obs["agentview_image"], obs["robot0_eye_in_hand_image"]], 2).transpose(
                1, 2, 0
            )
        )
        while not np.all(all_done):
            act, log = policy.sample(
                prior,
                solver="ddpm",
                sample_steps=sampling_steps,
                condition_cfg={
                    "pointcloud": pointcloud,
                    "language_embedding": lang_emb,
                    "language_mask": lang_mask,
                },
                w_cfg=1.0,
            )
            # act = act.cpu().numpy()
            act = normalizer["action"].unnormalize(act.cpu().numpy())

            for i in range(num_act_exec):
                next_obs, rew, done, _ = env.step(act[0, i])
                all_done = np.logical_or(all_done, done)
                all_rew += rew
                frames.append(
                    np.concatenate(
                        [next_obs["agentview_image"], next_obs["robot0_eye_in_hand_image"]], 2
                    ).transpose(1, 2, 0)
                )

                if i >= num_act_exec - To:
                    this_pointcloud = torch.tensor(
                        next_obs["agentview_pointcloud"], device=device, dtype=torch.float32
                    )
                    pointcloud[:, i - num_act_exec + To] = this_pointcloud[None]

                if np.all(all_done):
                    break

            print("Rewards:", all_rew)

        writer = imageio.get_writer(default_root_dir / "video.mp4", fps=30)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        env.close()

# import plotly.graph_objects as go

# # plot 3d
# pcd = this_pointcloud.cpu().numpy()
# fig = go.Figure()
# fig.add_trace(go.Scatter3d(x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], mode="markers", marker=dict(size=1)))
# fig.show()
