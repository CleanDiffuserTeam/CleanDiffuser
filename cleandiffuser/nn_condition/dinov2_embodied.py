from typing import Optional

import einops
import torch
import torch.nn as nn
import torchvision.transforms as T

from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.utils import Transformer, UntrainablePositionalEmbedding, generate_causal_mask, report_parameters


class RGBCondition(BaseNNCondition):
    """Prototype v1 for embodied perception.

    V1 accepts the following observations:
    - multi-view temporal RGBs (224 x 224 in [0, 1])

    For RGBs, pretrained DINOv2 models are used to extract features.
    - "facebook/dinov2-small" (384 embedding dimensions)
    - "facebook/dinov2-base" (768 embedding dimensions)
    - "facebook/dinov2-large" (1024 embedding dimensions)
    - "facebook/dinov2-giant" (1536 embedding dimensions)

    Before passing the model, the images are augmented with the following transformations:
    - For 3rd person views: Resize(224 / crop_rate), RandomCrop(224)
    - For 1st person views: Resize(224)
    - For all: Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    The input is supposed to be a dictionary with the following keys:
    - "FirstPersonRGBs": (B, V, T, C, H, W)
    - "ThirdPersonRGBs": (B, V, T, C, H, W)
    """

    def __init__(
        self,
        dinov2_path: str = "facebook/dinov2-base",
        freeze_image_encoder: bool = True,
        emb_dim: int = 768,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        first_person_views: int = 8,
        third_person_views: int = 8,
        max_history: int = 16,
        crop_rate: float = 0.9,
    ):
        super().__init__()

        # --- Image Tokenizer: DINOv2 ---
        from transformers import AutoModel

        self.dinov2 = AutoModel.from_pretrained(dinov2_path)
        dinov2_dims = self.dinov2.config.hidden_size
        if freeze_image_encoder:
            for param in self.dinov2.parameters():
                param.requires_grad = False
        self.transforms = {
            "resize": T.Resize(224),
            "normalize": T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            "random_crop": T.Compose([T.Resize(int(224 / crop_rate)), T.RandomCrop(224)]),
            "center_crop": T.Compose([T.Resize(int(224 / crop_rate)), T.CenterCrop(224)]),
        }

        # --- Type & Positional Embeddings ---
        self.history_emb = nn.Parameter(torch.randn(1, max_history, d_model))
        self.first_person_emb = nn.Parameter(torch.randn(1, first_person_views, d_model))
        self.third_person_emb = nn.Parameter(torch.randn(1, third_person_views, d_model))
        self.readout_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_emb = UntrainablePositionalEmbedding(d_model, max_positions=1000)

        # --- Fusion Transformer ---
        self.in_layer_RGB = nn.Linear(dinov2_dims, d_model, bias=False)
        # READOUT | RGBs
        num_tokens = 1 + first_person_views * max_history + third_person_views * max_history
        mask = generate_causal_mask(num_tokens)
        mask[:, 0] = 1.0
        mask[0, 1:] = 0.0
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.fusion_transformer = Transformer(
            d_model=d_model, nhead=nhead, num_layers=num_layers, attn_dropout=0.0, ffn_dropout=0.0, bias=False
        )
        self.out_layer = nn.Sequential(nn.GELU(), nn.Linear(d_model, emb_dim))

    def forward(self, condition: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # --- Image Processing ---
        first_person_rgbs = condition["FirstPersonRGBs"]  # (B, V, T, 3, 224, 224)
        B, V1, T1 = first_person_rgbs.shape[:3]
        first_person_rgbs = einops.rearrange(first_person_rgbs, "b v t c h w -> (b v t) c h w")
        first_person_rgbs = self.transforms["resize"](first_person_rgbs)
        first_person_rgbs = self.transforms["normalize"](first_person_rgbs)
        first_person_features = self.in_layer_RGB(self.dinov2(first_person_rgbs)["pooler_output"])
        first_person_features = einops.rearrange(first_person_features, "(b v t) d -> b v t d", v=V1, t=T1)
        first_person_features += self.first_person_emb.unsqueeze(2)
        first_person_features += self.history_emb.unsqueeze(1)
        first_person_features = einops.rearrange(first_person_features, "b v t d -> b (v t) d")

        third_person_rgbs = condition["ThirdPersonRGBs"]  # (B, V, T, 3, 224, 224)
        B, V2, T2 = third_person_rgbs.shape[:3]
        third_person_rgbs = einops.rearrange(third_person_rgbs, "b v t c h w -> (b v t) c h w")
        if self.training:
            third_person_rgbs = self.transforms["random_crop"](third_person_rgbs)
        else:
            third_person_rgbs = self.transforms["center_crop"](third_person_rgbs)
        third_person_rgbs = self.transforms["normalize"](third_person_rgbs)
        third_person_features = self.in_layer_RGB(self.dinov2(third_person_rgbs)["pooler_output"])
        third_person_features = einops.rearrange(third_person_features, "(b v t) d -> b v t d", v=V2, t=T2)
        third_person_features += self.third_person_emb.unsqueeze(2)
        third_person_features += self.history_emb.unsqueeze(1)
        third_person_features = einops.rearrange(third_person_features, "b v t d -> b (v t) d")

        # --- Assemble Tokens ---
        x = torch.cat([self.readout_emb.repeat(B, 1, 1), first_person_features, third_person_features], dim=1)
        x += self.positional_emb(torch.arange(x.shape[1], device=x.device))

        y, _ = self.fusion_transformer(x, mask=self.mask)

        return self.out_layer(y[:, 0])


if __name__ == "__main__":
    device = "cuda:5"

    x = {
        "FirstPersonRGBs": torch.rand(2, 1, 2, 3, 300, 300, device=device),
        "ThirdPersonRGBs": torch.rand(2, 1, 2, 3, 400, 400, device=device),
    }

    model = RGBCondition(
        dinov2_path="/home/dzb/pretrained_models/dinov2/dinov2-base",
        freeze_image_encoder=True,
        d_model=768,
        nhead=12,
        num_layers=12,
        first_person_views=1,
        third_person_views=1,
        max_history=2,
        crop_rate=0.9,
    ).to(device)

    report_parameters(model)

    y = model(x)

    print(y.shape)
