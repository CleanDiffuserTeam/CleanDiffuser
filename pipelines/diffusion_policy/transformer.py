from pathlib import Path
from typing import Dict, List, Optional

import einops
import gym
import h5py
import numpy as np
import pytorch_lightning as L
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import VisionTransformer
from transformers import AutoProcessor, T5EncoderModel, T5Tokenizer, ViTConfig, ViTModel

from cleandiffuser.dataset.libero_dataset import LiberoDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.env import libero
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import DiT1dV2, DiT1dWithACICrossAttention
from cleandiffuser.utils import PositionalEmbedding, set_seed


def pad(token):
    max_len = max([len(each) for each in token]) + 1
    return np.stack(
        [np.pad(each, (0, max_len - len(each)), "constant", constant_values=2048) for each in token]
    )


class T5LanguageEncoder:
    """T5LanguageEncoder

    A wrapped T5 encoder for convinient language encoding.
    It accepts a list of sentences and returns the feature tensor and padding mask.
    The length of the output tensor equals to the length of the longest sentence.

    Args:
        pretrained_model_name_or_path: The name or path of the pretrained model.
        device: The device to run the model on.

    Examples:
        >>> lang_encoder = T5LanguageEncoder()
        >>> h, mask = lang_encoder(["Hello world!", "How are you?"])
        >>> print(h.shape, mask.shape)
        torch.Size([2, 4, 768]) torch.Size([2, 4])
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "google-t5/t5-base",
        device: str = "cpu",
    ):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = T5EncoderModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

    @torch.no_grad()
    def encode(self, sentences: List[str]):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        outputs = self.model(input_ids=input_ids.to(self.device))
        last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs.attention_mask
        return last_hidden_states, attention_mask.to(last_hidden_states.device)

    def __call__(self, sentences: List[str]):
        return self.encode(sentences)


class ViTAndT5VisionLanguageCondition(BaseNNCondition):
    def __init__(
        self,
        vit_pretrained_model_name_or_path: str = "google/vit-base-patch16-224-in21k",
        vit_feat_dim: int = 384,
        t5_pretrained_model_name_or_path: str = "google-t5/t5-base",
        t5_feat_dim: int = 384,
        freeze: bool = False,
        To: int = 2,
    ):
        super().__init__()
        if vit_pretrained_model_name_or_path:
            self.vit_model = ViTModel.from_pretrained(vit_pretrained_model_name_or_path)
            if freeze:
                for p in self.vit_model.parameters():
                    p.requires_grad_(False)
        else:
            config = ViTConfig()
            self.vit_model = ViTModel(config)

        self.vit_adapter = nn.Sequential(
            nn.Linear(768, vit_feat_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(vit_feat_dim, vit_feat_dim),
            nn.LayerNorm(vit_feat_dim),
        )
        self.vit_model.pooler.requires_grad_(False)

        self.tokenizer = T5Tokenizer.from_pretrained(t5_pretrained_model_name_or_path)
        self.t5_model = (
            T5EncoderModel.from_pretrained(t5_pretrained_model_name_or_path)
            .eval()
            .requires_grad_(False)
        )
        self.t5_adapter = nn.Sequential(
            nn.Linear(768, t5_feat_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(t5_feat_dim, t5_feat_dim),
            nn.LayerNorm(t5_feat_dim),
        )
        self._ignored_hparams.append("t5_model")

        self.To_pos_emb = nn.Parameter(torch.randn(1, To, 1, vit_feat_dim) * 0.02)

    @torch.no_grad()
    def language_encode(self, sentences: List[str]):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        outputs = self.t5_model(input_ids=input_ids.to(self.t5_model.device))
        last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs.attention_mask
        return last_hidden_states, attention_mask.to(last_hidden_states.device)

    def forward(self, condition: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None):
        image = condition["image"]
        b = image.shape[0]

        language = condition.get("language", None)
        language_embedding = condition.get("language_embedding", None)
        language_mask = condition.get("language_mask", None)
        assert language is not None or language_embedding is not None, (
            "Either language or language_embedding and language_mask must be provided."
        )
        if language is not None:
            language_embedding, language_mask = self.language_encode(language)
        # transform to pytorch attention padding mask, where True means padding.
        if language_mask is not None:
            language_mask = torch.logical_not(language_mask.to(torch.bool))

        image = einops.rearrange(image, "b To C H W -> (b To) C H W")
        vision_feat = self.vit_model(image)["last_hidden_state"][:, 1:]  # (b * To, 196, 768)
        vision_feat = self.vit_adapter(vision_feat)
        vision_feat = einops.rearrange(vision_feat, "(b To) n d -> b To n d", b=b)
        vision_feat = vision_feat + self.To_pos_emb
        vision_feat = einops.rearrange(vision_feat, "b To n d -> b (To n) d")

        lang_feat = self.t5_adapter(language_embedding)

        return {
            "vec_condition": None,
            "lang_condition": lang_feat,
            "lang_condition_mask": language_mask,
            "vis_condition": vision_feat,
            "vis_condition_mask": None,
        }


class SDPAAttention(nn.Module):
    def __init__(self, hidden_size: int = 384, num_heads: int = 6):
        super().__init__()
        self.nhead = num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, is_causal=False):
        q_emb = einops.rearrange(self.q_proj(q), "b t (h d) -> b h t d", h=self.nhead)
        k_emb = einops.rearrange(self.k_proj(k), "b t (h d) -> b h t d", h=self.nhead)
        v_emb = einops.rearrange(self.v_proj(v), "b t (h d) -> b h t d", h=self.nhead)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q_emb, k_emb, v_emb, attn_bias, is_causal=is_causal
        )
        attn = einops.rearrange(attn, "b h t d -> b t (h d)")
        return attn


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 6,
    ):
        super().__init__()
        self.sa_norm = nn.LayerNorm(hidden_size)
        self.self_attn = SDPAAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.ca_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = SDPAAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x, c, attn_bias=None):
        # self attention
        h = self.sa_norm(x)
        x = x + self.self_attn(h, h, h, is_causal=True)

        # cross attention
        h = self.ca_norm(x)
        x = x + self.cross_attn(h, c, c, attn_bias=attn_bias)

        # ffn
        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        return x
    

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 6,
    ):
        super().__init__()
        self.sa_norm = nn.LayerNorm(hidden_size)
        self.self_attn = SDPAAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x, attn_bias=None):
        # self attention
        h = self.sa_norm(x)
        x = x + self.self_attn(h, h, h, is_causal=True)

        # ffn
        h = self.ffn_norm(x)
        x = x + self.ffn(h)
        return x


class TransformerActionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
    ):
        super().__init__()
        self.tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        self.vocab_size = 2048 + 1  # FAST + EOS
        self.vocab_emb = nn.Embedding(self.vocab_size, hidden_size)
        self.bos_emb = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.transformer = nn.ModuleList(
            [
                TransformerDecoderBlock(hidden_size=hidden_size, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )
        self.pos_emb = PositionalEmbedding(hidden_size, max_positions=256)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.vocab_size),
        )

    def forward(self, action: torch.Tensor, condition: torch.Tensor):
        token = self.tokenizer(action.cpu().numpy())
        token = torch.tensor(pad(token), device=action.device)
        b, t = token.shape
        emb = self.vocab_emb(token)
        emb = torch.cat([self.bos_emb.expand(b, 1, -1), emb[:, :-1]], dim=1)
        x = emb + self.pos_emb(torch.arange(t, device=emb.device))[None]
        for i, layer in enumerate(self.transformer):
            x = layer(x, condition["lang_condition"] if i % 2 == 0 else condition["vis_condition"])
        return self.out_proj(x), token

    def auto_regression(self, condition: torch.Tensor):
        b = condition.shape[0]
        emb = self.bos_emb.expand(b, 1, -1)
        t = 1
        tokens = []
        for _ in range(64):
            x = emb + self.pos_emb(torch.arange(t, device=emb.device))[None]
            for layer in self.transformer:
                x = layer(x, condition)
            logits = self.out_proj(x)
            next_token = torch.argmax(logits, dim=-1)[:, -1:]
            next_emb = self.vocab_emb(next_token)
            emb = torch.cat([emb, next_emb], dim=1)
            t += 1
            tokens.append(next_token.clone())
        return torch.stack(tokens, dim=1)


class Policy(L.LightningModule):
    def __init__(self, hidden_size: int = 384, num_heads: int = 6, num_layers: int = 12):
        super().__init__()
        self.condition_encoder = ViTAndT5VisionLanguageCondition(
            # vit_pretrained_model_name_or_path="google/vit-base-patch16-224-in21k",
            vit_pretrained_model_name_or_path=None,
            vit_feat_dim=hidden_size,
            t5_pretrained_model_name_or_path="google-t5/t5-base",
            t5_feat_dim=hidden_size,
            To=2,
        )
        self.policy_head = TransformerActionHead(
            hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers
        )

    def forward(self, action: torch.Tensor, condition: Dict[str, torch.Tensor]):
        condition = self.condition_encoder(condition)
        logits, token = self.policy_head(action, condition)
        return logits, token

    def loss(self, action: torch.Tensor, condition: Dict[str, torch.Tensor]):
        logits, token = self.forward(action, condition)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, 2049),
            torch.nn.functional.one_hot(token, 2049).float().reshape(-1, 2049),
        )

    def training_step(self, batch):
        action, condition = batch["x"], batch["condition"]
        loss = self.loss(action, condition)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        x = "vit_model"
        m = self.condition_encoder
        return torch.optim.Adam(
            [
                {"params": [p for n, p in m.named_parameters() if x not in n], "lr": 3e-4},
                {"params": [p for n, p in m.named_parameters() if x in n], "lr": 3e-4},
            ]
        )


class DatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
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
        # image_ego = item["observation"]["color_ego"].astype(np.float32) / 255.0
        # image_ego = self.normalize(torch.tensor(image_ego))
        # Concatenate two-view images along the time dimension
        # combined_image = torch.cat([image, image_ego], dim=0)
        combined_image = image

        act = item["action"]
        obs = {
            "image": combined_image,
            "language_embedding": item["language_embedding"],
            "language_mask": item["language_mask"],
        }
        return {"x": act, "condition": obs}


task_suite = "libero_goal"
seed = 0
dataset_path = Path(__file__).parents[2] / f"dev/libero/{task_suite}.zarr"
devices = [0, 1, 2, 3]  # List of GPU ids to train on, cuda:0 for default
default_root_dir = Path(__file__).parents[2] / f"ab_results/transformer/{task_suite}/"
training_steps = 30_000
save_every_n_steps = 10_000
ckpt_file = "epoch=60-step=30000.ckpt"
env_name = "libero-goal-v0"
task_id = 9
sampling_steps = 20

if __name__ == "__main__":
    dataset = LiberoDataset("dev/libero/libero_goal.zarr", observation_meta=["color"], To=2, Ta=20)
    dataset = DatasetWrapper(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=8, persistent_workers=True
    )

    policy = Policy()
    callback = ModelCheckpoint(
        dirpath=default_root_dir,
        every_n_train_steps=save_every_n_steps,
        save_top_k=-1,
    )
    trainer = L.Trainer(
        devices=devices,
        max_steps=training_steps,
        callbacks=[callback],
        accumulate_grad_batches=1,
        default_root_dir=default_root_dir,
    )
    trainer.fit(policy, dataloader)
