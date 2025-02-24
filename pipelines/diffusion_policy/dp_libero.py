from pathlib import Path
from typing import Dict, Optional

import einops
import gym
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint
from termcolor import cprint
from transformers import T5Config, ViTConfig, ViTModel

from cleandiffuser.dataset.libero_dataset import LiberoDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.env import libero  # noqa
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import DiT1dWithACICrossAttention
from cleandiffuser.utils import UntrainablePositionalEmbedding, set_seed
from cleandiffuser.utils.t5 import T5LanguageEncoder


class ViTAndT5VisionLanguageCondition(BaseNNCondition):
    """ViT and T5 Vision-Language Condition.

    A vision-language condition that uses a pretrained (optional) Vision Transformer (ViT) and a pretrained
    T5 model to encode the vision and language inputs, respectively. The encoded vision and language embeddings
    are then used to condition the diffusion model.

    Args:
        vit_pretrained_model_name_or_path (str):
            The ViT model name or path to load the pretrained model from. If None, a new ViT model will be created.
        t5_hidden_dim (int):
            The hidden dimension of the T5 model.
        t5_max_seq_len (int):
            The maximum sequence length of the T5 model.
        emb_dim (int):
            The embedding dimension of the encoded vision and language embeddings.
        freeze (bool):
            Whether to freeze the ViT model or not.
        To (int):
            The number of time steps to condition on.
        n_views (int):
            The number of views to condition on.
    """

    def __init__(
        self,
        vit_pretrained_model_name_or_path: str = "google/vit-base-patch16-224-in21k",
        t5_hidden_dim: int = 768,
        t5_max_seq_len: int = 32,
        emb_dim: int = 384,
        freeze: bool = False,
        To: int = 2,
        n_views: int = 2,
    ):
        super().__init__()
        self.To = To
        self.n_views = n_views

        # load pretrained ViT model or create a new one
        if vit_pretrained_model_name_or_path:
            self.vit_model = ViTModel.from_pretrained(vit_pretrained_model_name_or_path)
            if freeze:
                for p in self.vit_model.parameters():
                    p.requires_grad_(False)
        else:
            config = ViTConfig()
            self.vit_model = ViTModel(config)
        vit_hidden_dim = self.vit_model.config.hidden_size

        self.vit_adapter = nn.Sequential(
            nn.Linear(vit_hidden_dim, emb_dim), nn.GELU(approximate="tanh")
        )
        self.vit_model.pooler.requires_grad_(False)

        self.t5_adapter = nn.Sequential(
            nn.Linear(t5_hidden_dim, emb_dim), nn.GELU(approximate="tanh")
        )

        self.To_pos_emb = nn.Parameter(torch.randn(1, To * n_views, 1, emb_dim) * 0.02)

        num_patches = (self.vit_model.config.image_size // self.vit_model.config.patch_size) ** 2
        vit_pos_emb = (
            UntrainablePositionalEmbedding(emb_dim, 1000)(torch.arange(num_patches))[None, None]
            * 0.2
        )
        self.vit_pos_emb = nn.Parameter(vit_pos_emb)
        t5_pos_emb = (
            UntrainablePositionalEmbedding(emb_dim, 100)(torch.arange(t5_max_seq_len))[None] * 0.2
        )
        self.t5_pos_emb = nn.Parameter(t5_pos_emb)

    def forward(self, condition: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None):
        image = condition["image"]  # (b, n_views, To, C, H, W)
        b = image.shape[0]

        language_embedding = condition["language_embedding"]
        language_mask = condition["language_mask"]

        # transform to pytorch attention padding mask, where True means padding.
        if language_mask is not None:
            language_mask = torch.logical_not(language_mask.to(torch.bool))

        image = einops.rearrange(image, "b To C H W -> (b To) C H W")
        vision_feat = self.vit_model(image)["last_hidden_state"][:, 1:]  # (b * To, 196, 768)
        vision_feat = self.vit_adapter(vision_feat)
        vision_feat = einops.rearrange(vision_feat, "(b To) m d -> b To m d", b=b)
        vision_feat = vision_feat + self.To_pos_emb + self.vit_pos_emb
        vision_feat = einops.rearrange(vision_feat, "b To m d -> b (To m) d")

        lang_feat = self.t5_adapter(language_embedding) + self.t5_pos_emb

        return {
            "vec_condition": None,
            "lang_condition": lang_feat,
            "lang_condition_mask": language_mask,
            "vis_condition": vision_feat,
            "vis_condition_mask": None,
        }


class DatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
        # add normalization and random crop to 3rd person view
        # add only normalization to egocentric view
        self.normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        self.vis_aug = T.Compose(
            [
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                T.RandomCrop(200),
                T.Resize(224),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Rescale to [0, 1] and apply random crop and resize
        image = item["observation"]["color"].astype(np.float32) / 255.0
        image = self.vis_aug(torch.tensor(image))
        image_ego = item["observation"]["color_ego"].astype(np.float32) / 255.0
        image_ego = self.normalize(torch.tensor(image_ego))
        # Concatenate two-view images along the time dimension
        combined_image = torch.cat([image, image_ego], dim=0)

        act = item["action"]
        obs = {
            "image": combined_image,
            "language_embedding": item["language_embedding"],
            "language_mask": item["language_mask"],
        }
        return {"x0": act, "condition_cfg": obs}


class FTContinuousDiffusionSDE(ContinuousDiffusionSDE):
    # Override the configure_optimizers method to use different learning rates for different parts of the model
    # 1/10 of the learning rate for the pretrained vision transformer
    def configure_optimizers(self):
        x = "vit_model"
        m = self.model
        return torch.optim.Adam(
            [
                {"params": [p for n, p in m.named_parameters() if x not in n], "lr": 1e-4},
                {"params": [p for n, p in m.named_parameters() if x in n], "lr": 1e-5},
            ]
        )


# --- Config ---
task_suite = "libero_spatial"
t5_pretrained_model_name_or_path = "google-t5/t5-base"
vit_pretrained_model_name_or_path = "google/vit-base-patch16-224-in21k"
seed = 0
To = 1
Ta = 16
num_act_exec = 8
mode = "training"  # training, inference or rendering
dataset_path = Path(__file__).parents[2] / f"dev/libero/{task_suite}.zarr"
devices = [0, 1, 2, 3]  # List of GPU ids to train on, cuda:0 for default
default_root_dir = Path(__file__).parents[2] / f"results/diffusion_policy/{task_suite}/"
training_steps = 100_000
save_every_n_steps = 10_000
ckpt_file = "epoch=409-step=100000.ckpt"
env_name = "libero-spatial-v0"
task_id = 9
sampling_steps = 20

if __name__ == "__main__":
    set_seed(seed)

    # --- Dataset ---
    dataset = LiberoDataset(
        data_path=dataset_path,
        observation_meta=["color", "color_ego"],
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

    nn_diffusion = DiT1dWithACICrossAttention(
        x_dim=act_dim,
        x_seq_len=Ta,
        emb_dim=768,
        d_model=384,
        n_heads=6,
        depth=12,
        timestep_emb_type="untrainable_fourier",
        timestep_emb_params={"scale": 0.2},
    )
    nn_condition = ViTAndT5VisionLanguageCondition(
        vit_pretrained_model_name_or_path=vit_pretrained_model_name_or_path,
        t5_hidden_dim=t5_hidden_dim,
        t5_max_seq_len=32,
        emb_dim=768,
        freeze=False,
        To=To,
        n_views=2,
    )

    policy = FTContinuousDiffusionSDE(
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
            strategy="ddp_find_unused_parameters_true",
        )
        trainer.fit(policy, dataloader)

    # -- Inference --
    elif mode == "inference":
        t5_language_encoder = T5LanguageEncoder(
            pretrained_model_name_or_path=t5_pretrained_model_name_or_path,
            max_length=32,
            device=f"cuda:{devices[0]}",
        )
        import imageio

        device = f"cuda:{devices[0]}"

        env = gym.make(
            env_name,
            task_id=task_id,
            image_size=224,
            require_depth=False,
            require_point_cloud=False,
            seed=seed,
        )
        cprint(f"TASK: {env.task_description}", "green", attrs=["bold"])

        lang_emb, lang_mask = t5_language_encoder([env.task_description])

        normalizer = dataset.get_normalizer()
        img_norm = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        center_crop = T.Compose(
            [
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                T.CenterCrop(200),
                T.Resize(224),
            ]
        )

        policy.load_state_dict(
            torch.load(default_root_dir / ckpt_file, map_location=device)["state_dict"]
        )
        policy = policy.to(device).eval()

        prior = torch.zeros((1, Ta, act_dim), device=device)

        success = []

        for init_state_id in range(env.num_init_states):
            dummy_steps = 0
            obs, all_done, all_rew = env.reset(init_state_id=init_state_id), False, 0

            image = torch.tensor(obs["agentview_image"] / 255.0, device=device, dtype=torch.float32)
            image = center_crop(image)
            image = image[None, None].repeat(1, To, 1, 1, 1)
            image_ego = torch.tensor(
                obs["robot0_eye_in_hand_image"] / 255.0, device=device, dtype=torch.float32
            )
            image_ego = img_norm(image_ego)
            image_ego = image_ego[None, None].repeat(1, To, 1, 1, 1)
            image = torch.cat([image, image_ego], dim=1)

            frames = []
            frames.append(
                np.concatenate(
                    [obs["agentview_image"], obs["robot0_eye_in_hand_image"]], 2
                ).transpose(1, 2, 0)
            )

            while not np.all(all_done):
                act, log = policy.sample(
                    prior,
                    solver="ddpm",
                    sample_steps=sampling_steps,
                    condition_cfg={
                        "image": image,
                        "language_embedding": lang_emb,
                        "language_mask": lang_mask,
                    },
                    use_ema=False,
                    w_cfg=1.0,
                )
                act = normalizer["action"].unnormalize(act.cpu().numpy())

                # Objects may fall down from sky when initializing the environment
                # Take a few dummy steps to stabilize the environment
                if dummy_steps < 2:
                    act = np.zeros((1, num_act_exec, act_dim), dtype=np.float32)
                    dummy_steps += 1

                for i in range(num_act_exec):
                    next_obs, rew, done, _ = env.step(act[0, i])
                    all_done = np.logical_or(all_done, done)
                    all_rew += rew

                    if i >= num_act_exec - To:
                        this_image = torch.tensor(
                            next_obs["agentview_image"] / 255.0, device=device, dtype=torch.float32
                        )
                        this_image = this_image[None]
                        this_image = center_crop(this_image)
                        this_image_ego = torch.tensor(
                            next_obs["robot0_eye_in_hand_image"] / 255.0,
                            device=device,
                            dtype=torch.float32,
                        )
                        this_image_ego = this_image_ego[None]
                        this_image_ego = img_norm(this_image_ego)

                        image[:, i - num_act_exec + To] = this_image
                        image[:, i - num_act_exec + To * 2] = this_image_ego

                    if np.all(all_done):
                        break

                    frames.append(
                        np.concatenate(
                            [next_obs["agentview_image"], next_obs["robot0_eye_in_hand_image"]], 2
                        ).transpose(1, 2, 0)
                    )

                cprint(f"[Test {init_state_id}] Success: {all_rew}", "green")

            writer = imageio.get_writer(default_root_dir / "video.mp4", fps=30)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

            if all_rew:
                success.append(True)

        print(f"Success rate: {np.sum(success) / env.num_init_states}")
