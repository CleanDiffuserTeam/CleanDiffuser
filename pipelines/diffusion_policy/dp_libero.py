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
from transformers import T5EncoderModel, T5Tokenizer

from cleandiffuser.dataset.libero_dataset import LiberoDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.env import libero
from cleandiffuser.nn_diffusion import DiT1dWithACICrossAttention
from cleandiffuser.utils import set_seed


class ViTAndT5VisionLanguageCondition(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        vit_d_model: int = 384,
        vit_num_heads: int = 6,
        vit_depth: int = 12,
        t5_pretrained_model_name_or_path: str = "google-t5/t5-base",
        t5_feat_dim: int = 384,
    ):
        super().__init__()
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
            embed_dim=vit_d_model,
            depth=vit_depth,
            num_heads=vit_num_heads,
            class_token=False,
        )
        self.tokenizer = T5Tokenizer.from_pretrained(t5_pretrained_model_name_or_path)
        self.model = (
            T5EncoderModel.from_pretrained(t5_pretrained_model_name_or_path)
            .eval()
            .requires_grad_(False)
        )

        self.t5_proj = nn.Sequential(
            nn.Linear(768, t5_feat_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(t5_feat_dim, t5_feat_dim),
        )

    @torch.no_grad()
    def language_encode(self, sentences: List[str]):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        outputs = self.model(input_ids=input_ids.to(self.model.device))
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states, inputs.attention_mask

    def forward(self, condition: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None):
        image = condition["image"]
        language = condition.get("language", None)
        language_embedding = condition.get("language_embedding", None)
        language_mask = condition.get("language_mask", None)
        assert language is not None or (
            language_embedding is not None and language_mask is not None
        ), "Either language or language_embedding and language_mask must be provided."

        if language is not None:
            language_embedding, language_mask = self.language_encode(language)

        language_mask = torch.logical_not(language_mask.to(torch.bool))

        vision_feat = self.vit(einops.rearrange(image, "b t c h w -> (b t) c h w"))
        vision_feat = einops.rearrange(vision_feat, "(b t) n d -> b (t n) d", b=image.shape[0])

        lang_feat = self.t5_proj(language_embedding)

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

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        act = item["action"]
        obs = {
            "image": item["observation"]["color"].astype(np.float32) / 255.0,
            "language_embedding": item["language_embedding"],
            "language_mask": item["language_mask"],
        }
        return {"x0": act, "condition_cfg": obs}


# --- Config ---
task_suite = "libero_goal"
t5_pretrained_model_name_or_path = "/home/dzb/pretrained/t5-base"
seed = 0
To = 2
Ta = 16
num_act_exec = 8
mode = "training"  # training, inference or rendering
dataset_path = Path(__file__).parents[2] / f"dev/libero/{task_suite}.zarr"
devices = [3]  # List of GPU ids to train on, cuda:0 for default
default_root_dir = Path(__file__).parents[2] / f"results/diffusion_policy/libero_{task_suite}/"
training_steps = 500_000
save_every_n_steps = 50_000
ckpt_file = "step=500000.ckpt"
sampling_steps = 20

if __name__ == "__main__":
    set_seed(seed)

    # --- Dataset ---
    dataset = LiberoDataset(data_path=dataset_path, observation_meta=["color"], To=To, Ta=Ta)
    act_dim = 7

    dataloader = torch.utils.data.DataLoader(
        DatasetWrapper(dataset),
        batch_size=64,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    # --- Model ---
    nn_diffusion = DiT1dWithACICrossAttention(
        x_dim=act_dim,
        x_seq_len=Ta,
        emb_dim=384,
        d_model=384,
        n_heads=6,
        depth=12,
        timestep_emb_type="untrainable_fourier",
    )
    nn_condition = ViTAndT5VisionLanguageCondition(
        img_size=224,
        patch_size=16,
        in_chans=3,
        vit_d_model=384,
        vit_num_heads=6,
        vit_depth=6,
        t5_pretrained_model_name_or_path=t5_pretrained_model_name_or_path,
        t5_feat_dim=384,
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
