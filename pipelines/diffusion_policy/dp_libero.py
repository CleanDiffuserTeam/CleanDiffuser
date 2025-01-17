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
from transformers import T5EncoderModel, T5Tokenizer

from cleandiffuser.dataset.libero_dataset import LiberoDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.env import libero
from cleandiffuser.nn_diffusion import DiT1dV2, DiT1dWithACICrossAttention
from cleandiffuser.utils import set_seed


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


class ViTAndT5VisionLanguageCondition(nn.Module):
    """Vision-Language Condition Network with ViT and T5 backbones.

    This condition network accepts tree types of conditions:
    - image:
        Provided in key "image".
        Supposed to be (B, To, c, h, w).
        It will be processed by a random initialized ViT.
    - language:
        Either provided in key "language" or ("language_embedding", "language_mask").
        If "language" is provided, it is supposed to be a list of strings and
        will be processed by a pretrained T5 model.
        If "language_embedding" and "language_mask" are provided, it is supposed to be in shape
        (B, L, D), (B, L) respectively.
    - lowdim:
        Provided in key "lowdim". Supposed to be (B, D).

    After processing, it returns a dictionary with the following keys:
    - vec_condition: lowdim feature         (B, lowdim_feat_dim)
    - lang_condition: language embedding    (B, lang_seq_len, t5_feat_dim)
    - lang_condition_mask: language mask    (B, lang_seq_len)
    - vis_condition: image embedding        (B, num_patches, vit_d_model)
    - vis_condition_mask: image mask        None

    Args:
        img_size (int):
            The size of the input image.
        patch_size (int):
            The size of the image patch.
        in_chans (int):
            The number of input channels.
        vit_d_model (int):
            The dimension of the ViT embedding.
        vit_num_heads (int):
            The number of heads in the ViT.
        vit_depth (int):
            The depth of the ViT.
        t5_pretrained_model_name_or_path (str):
            The name or path of the pretrained T5 model.
        t5_feat_dim (int):
            The dimension of the T5 embedding.
        To (int):
            The number of previous + current timesteps.
    """

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
        To: int = 2,
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
        self.lang_pos_emb = nn.Parameter(torch.randn(1, 32, t5_feat_dim) * 0.02)
        self.vis_pos_emb = nn.Parameter(torch.randn(1, 196, vit_d_model) * 0.02)
        self.To_pos_emb = nn.Parameter(torch.randn(1, To, 1, vit_d_model) * 0.02)

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
        assert language is not None or language_embedding is not None, (
            "Either language or language_embedding and language_mask must be provided."
        )

        if language is not None:
            language_embedding, language_mask = self.language_encode(language)

        # transform to pytorch attention padding mask, where True means padding.
        if language_mask is not None:
            language_mask = torch.logical_not(language_mask.to(torch.bool))

        vision_feat = self.vit(einops.rearrange(image, "b t c h w -> (b t) c h w"))
        vision_feat = vision_feat + self.vis_pos_emb
        vision_feat = einops.rearrange(vision_feat, "(b t) n d -> b t n d", b=image.shape[0])
        vision_feat = vision_feat + self.To_pos_emb
        vision_feat = einops.rearrange(vision_feat, "b t n d -> b (t n) d")

        lang_feat = self.t5_proj(language_embedding)
        lang_feat = lang_feat + self.lang_pos_emb

        return {
            "vec_condition": None,
            "lang_condition": lang_feat,
            "lang_condition_mask": language_mask,
            "vis_condition": vision_feat,
            "vis_condition_mask": None,
        }


class ResnetAndT5VisionLanguageCondition(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        resnet_feat_dim: int = 512,
        pretrained: bool = True,
        freeze: bool = False,
        t5_pretrained_model_name_or_path: str = "google-t5/t5-base",
        emb_dim: int = 384,
        To: int = 2,
    ):
        super().__init__()
        self.resnet = timm.create_model(model_name, pretrained=pretrained)
        if freeze:
            self.resnet = self.resnet.requires_grad_(False).eval()
        self.resnet.fc.requires_grad_(False)

        self.tokenizer = T5Tokenizer.from_pretrained(t5_pretrained_model_name_or_path)
        self.t5_model = T5EncoderModel.from_pretrained(t5_pretrained_model_name_or_path)
        self.t5_model = self.t5_model.requires_grad_(False).eval()

        self.img_proj = nn.Sequential(
            nn.Linear(resnet_feat_dim * To, emb_dim // 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(emb_dim // 2, emb_dim // 2),
        )
        self.lang_proj = nn.Sequential(
            nn.Linear(768, emb_dim // 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(emb_dim // 2, emb_dim // 2),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def language_encode(self, sentences: List[str]):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        outputs = self.t5_model(input_ids=input_ids.to(self.device))
        last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs.attention_mask
        return last_hidden_states, attention_mask.to(last_hidden_states.device)

    def forward(self, condition: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None):
        image = condition["image"]
        language = condition.get("language", None)
        language_embedding = condition.get("language_embedding", None)
        language_mask = condition.get("language_mask", None)
        assert language is not None or language_embedding is not None, (
            "Either language or language_embedding and language_mask must be provided."
        )
        if language is not None:
            language_embedding, language_mask = self.language_encode(language)

        # image encoding
        leading_dims = image.shape[:-3]
        image = image.reshape(-1, *image.shape[-3:])
        img_feat = self.resnet.forward_features(image)
        img_feat = self.resnet.global_pool(img_feat)
        img_feat = img_feat.reshape(*leading_dims, -1)
        img_feat = torch.flatten(img_feat, start_dim=1)
        img_feat = self.img_proj(img_feat)

        # language encoding
        lang_feat = (language_embedding * language_mask[:, :, None]).sum(1)
        lang_feat = self.lang_proj(lang_feat)

        condition = torch.cat([img_feat, lang_feat], dim=-1)
        return condition


class DatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
        self.vis_aug = T.Compose(
            [
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
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
        # image_ego = self.vis_aug(torch.tensor(image_ego))
        # Combine two-view images as one 6-channel image
        # combined_image = torch.cat([image, image_ego], dim=1)

        act = item["action"]
        obs = {
            "image": image,
            "language_embedding": item["language_embedding"],
            "language_mask": item["language_mask"],
        }
        return {"x0": act, "condition_cfg": obs}


# --- Config ---
task_suite = "libero_goal"
t5_pretrained_model_name_or_path = "/home/dzb/pretrained/t5-base"
seed = 1
To = 2
Ta = 16
num_act_exec = 8
mode = "inference"  # training, inference or rendering
dataset_path = Path(__file__).parents[2] / f"dev/libero/{task_suite}.zarr"
devices = [3]  # List of GPU ids to train on, cuda:0 for default
default_root_dir = Path(__file__).parents[2] / f"results/diffusion_policy/{task_suite}/"
training_steps = 500_000
save_every_n_steps = 10_000
ckpt_file = "epoch=60-step=30000.ckpt"
env_name = "libero-goal-v0"
task_id = 0
sampling_steps = 20

if __name__ == "__main__":
    set_seed(seed)

    # --- Dataset ---
    dataset = LiberoDataset(
        data_path=dataset_path,
        observation_meta=["color"],
        To=To,
        Ta=Ta,
    )
    act_dim = 7
    lowdim_dim = 9

    dataloader = torch.utils.data.DataLoader(
        DatasetWrapper(dataset),
        batch_size=128,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    # --- Model ---
    nn_diffusion = DiT1dV2(
        x_dim=act_dim,
        x_seq_len=Ta,
        emb_dim=768,
        d_model=384,
        n_heads=6,
        depth=12,
        timestep_emb_type="untrainable_fourier",
        timestep_emb_params={"scale": 0.2},
    )
    nn_condition = ResnetAndT5VisionLanguageCondition(
        model_name="resnet18",
        resnet_feat_dim=512,
        pretrained=False,
        freeze=False,
        t5_pretrained_model_name_or_path=t5_pretrained_model_name_or_path,
        emb_dim=768,
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
        )
        trainer.fit(policy, dataloader)

    # -- Inference --
    elif mode == "inference":
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
        print(env.task_description)
        normalizer = dataset.get_normalizer()
        center_crop = T.Compose(
            [
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                T.CenterCrop(200),
                T.Resize(224),
            ]
        )

        policy.load_state_dict(torch.load(default_root_dir / ckpt_file)["state_dict"])
        policy = policy.to(device).eval()

        prior = torch.zeros((1, Ta, act_dim), device=device)

        success = []

        for init_state_id in range(env.num_init_states):
            dummy_steps = 0
            obs, all_done, all_rew = env.reset(init_state_id=init_state_id), False, 0

            image = torch.tensor(obs["agentview_image"] / 255.0, device=device, dtype=torch.float32)
            image = center_crop(image)
            image = image[None, None].repeat(1, To, 1, 1, 1)
            
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
                        "image": image,
                        "language": env.task_description,
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
                        image[:, i - num_act_exec + To] = this_image

                    if np.all(all_done):
                        break
                    
                    frames.append(
                        np.concatenate(
                            [next_obs["agentview_image"], next_obs["robot0_eye_in_hand_image"]], 2
                        ).transpose(1, 2, 0)
                    )

                print(f"[Test {init_state_id}] Success: {all_rew}")

            writer = imageio.get_writer(default_root_dir / "video.mp4", fps=30)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            
            if all_rew:
                success.append(True)

        print(f"Success rate: {np.mean(success)}")

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
            require_depth=False,
            require_point_cloud=False,
            seed=seed,
        )
        print(env.task_description)
        normalizer = dataset.get_normalizer()
        center_crop = T.Compose(
            [
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                T.CenterCrop(200),
                T.Resize(224),
            ]
        )

        policy.load_state_dict(torch.load(default_root_dir / ckpt_file)["state_dict"])
        policy = policy.to(device).eval()

        prior = torch.zeros((1, Ta, act_dim), device=device)
        obs, all_done, all_rew = env.reset(), False, 0

        image = torch.tensor(obs["agentview_image"] / 255.0, device=device, dtype=torch.float32)
        image = center_crop(image)
        image = image[None, None].repeat(1, To, 1, 1, 1)

        frames = []
        frames.append(
            np.concatenate([obs["agentview_image"], obs["robot0_eye_in_hand_image"]], 2).transpose(
                1, 2, 0
            )
        )
        dummy_steps = 0
        while not np.all(all_done):
            act, log = policy.sample(
                prior,
                solver="ddpm",
                sample_steps=sampling_steps,
                condition_cfg={
                    "image": image,
                    "language": env.task_description,
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
                frames.append(
                    np.concatenate(
                        [next_obs["agentview_image"], next_obs["robot0_eye_in_hand_image"]], 2
                    ).transpose(1, 2, 0)
                )

                if i >= num_act_exec - To:
                    this_image = torch.tensor(
                        next_obs["agentview_image"] / 255.0, device=device, dtype=torch.float32
                    )
                    this_image = this_image[None]
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
