from pathlib import Path
from typing import Dict, List, Optional

import einops
import gym
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import VisionTransformer
from transformers import T5EncoderModel, T5Tokenizer, ViTConfig, ViTModel

from cleandiffuser.dataset.libero_dataset import LiberoDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.env import libero
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import DiT1dV2, DiT1dWithACICrossAttention
from cleandiffuser.utils import at_least_ndim, set_seed


def dict_apply(x, fn):
    for k, v in x.items():
        if isinstance(v, dict):
            dict_apply(v, fn)
        else:
            x[k] = fn(v)
    return x


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
        )
        self._ignored_hparams.append("t5_model")

        self.To_pos_emb = nn.Parameter(torch.randn(1, To * 2, 1, vit_feat_dim) * 0.02)

        self.lowdim_adapter = nn.Sequential(
            nn.Linear(9, 384), nn.GELU(approximate="tanh"), nn.Linear(384, 384)
        )

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

        lowdim_feat = self.lowdim_adapter(condition["lowdim"][:, -1])

        return {
            "vec_condition": lowdim_feat,
            "lang_condition": lang_feat,
            "lang_condition_mask": language_mask,
            "vis_condition": vision_feat,
            "vis_condition_mask": None,
        }


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
        image_ego = item["observation"]["color_ego"].astype(np.float32) / 255.0
        image_ego = self.normalize(torch.tensor(image_ego))
        # Concatenate two-view images along the time dimension
        combined_image = torch.cat([image, image_ego], dim=0)
        # combined_image = image

        eef_state = item["observation"]["eef_states"]
        gripper_state = item["observation"]["gripper_states"]
        lowdim = np.concatenate([eef_state, gripper_state], axis=-1)

        act = item["action"]
        obs = {
            "image": combined_image,
            "language_embedding": item["language_embedding"],
            "language_mask": item["language_mask"],
            "lowdim": lowdim,
        }
        return {"x0": act, "condition_cfg": obs}


class FTContinuousDiffusionSDE(ContinuousDiffusionSDE):
    def configure_optimizers(self):
        x = "vit_model"
        m = self.model
        return torch.optim.Adam(
            [
                {"params": [p for n, p in m.named_parameters() if x not in n], "lr": 1e-4},
                {"params": [p for n, p in m.named_parameters() if x in n], "lr": 1e-4},
            ]
        )


# --- Config ---
task_suite = "libero_goal"
t5_pretrained_model_name_or_path = "google-t5/t5-base"
seed = 0
To = 2
Ta = 16
num_act_exec = 8
mode = "inference"  # training, inference or rendering
dataset_path = Path(__file__).parents[2] / f"dev/libero/{task_suite}.zarr"
devices = [3]  # List of GPU ids to train on, cuda:0 for default
default_root_dir = (
    Path(__file__).parents[2] / f"ab_results/diffusion_policy/{task_suite}_prenorm_lowdim/"
)
training_steps = 30_000
save_every_n_steps = 10_000
ckpt_file = "epoch=60-step=30000.ckpt"
env_name = "libero-goal-v0"
task_id = 9
sampling_steps = 20

# default_root_dir = (
#     Path(__file__).parents[2] / f"ab_results/diffusion_policy/{task_suite}_prenorm_singleview/"
# )
# ckpt_file = "epoch=60-step=30000.ckpt"
# nn_diffusion = DiT1dWithACICrossAttention(
#     x_dim=act_dim,
#     x_seq_len=Ta,
#     emb_dim=384,
#     d_model=384,
#     n_heads=6,
#     depth=12,
#     prenorm=True,
#     adaLN_on_cross_attn=True,
#     timestep_emb_type="untrainable_fourier",
#     timestep_emb_params={"scale": 0.2},
# )
# nn_condition = ViTAndT5VisionLanguageCondition(
#     # vit_pretrained_model_name_or_path="google/vit-base-patch16-224-in21k",
#     vit_pretrained_model_name_or_path=None,
#     vit_feat_dim=384,
#     t5_pretrained_model_name_or_path=t5_pretrained_model_name_or_path,
#     t5_feat_dim=384,
#     To=To,
# )

# policy = FTContinuousDiffusionSDE(
#     nn_diffusion=nn_diffusion,
#     nn_condition=nn_condition,
#     ema_rate=0.75,
#     x_max=torch.full((Ta, act_dim), 1.0),
#     x_min=torch.full((Ta, act_dim), -1.0),
# )
# policy.load_state_dict(
#     torch.load(default_root_dir / ckpt_file, map_location=device)["state_dict"]
# )
# policy = policy.to(device).eval()

# batch = next(iter(dataloader))
# batch = dict_apply(batch, lambda x:x.to(device))

# with torch.no_grad():
#     condition = nn_condition(batch['condition_cfg'])

if __name__ == "__main__":
    set_seed(seed)

    # --- Dataset ---
    dataset = LiberoDataset(
        data_path=dataset_path,
        observation_meta=["color", "color_ego", "eef_states", "gripper_states"],
        # observation_meta=["color"],
        To=To,
        Ta=Ta,
    )
    act_dim = 7
    lowdim_dim = 9

    dataloader = torch.utils.data.DataLoader(
        DatasetWrapper(dataset),
        batch_size=32,
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
        prenorm=True,
        adaLN_on_cross_attn=True,
        timestep_emb_type="untrainable_fourier",
        timestep_emb_params={"scale": 0.2},
    )
    nn_condition = ViTAndT5VisionLanguageCondition(
        # vit_pretrained_model_name_or_path="google/vit-base-patch16-224-in21k",
        vit_pretrained_model_name_or_path=None,
        vit_feat_dim=384,
        t5_pretrained_model_name_or_path=t5_pretrained_model_name_or_path,
        t5_feat_dim=384,
        To=To,
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
            accumulate_grad_batches=1,
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
            max_episode_steps=200,
        )
        print(env.task_description)
        normalizer = dataset.get_normalizer()
        normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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
            image_ego = normalize(image_ego)
            image_ego = image_ego[None, None].repeat(1, To, 1, 1, 1)
            image = torch.cat([image, image_ego], dim=1)
            
            lowdim = np.concatenate(
                [obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]
            )
            lowdim = torch.tensor(lowdim, device=device, dtype=torch.float32)[None, None]

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
                        "language": env.task_description,
                        "lowdim": lowdim,
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
                        this_image_ego = normalize(this_image_ego)

                        image[:, i - num_act_exec + To] = this_image
                        image[:, i - num_act_exec + To * 2] = this_image_ego
                        # image[:, i - num_act_exec + To] = this_image
                        
                        this_lowdim = np.concatenate(
                            [obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]
                        )
                        this_lowdim = torch.tensor(this_lowdim, device=device, dtype=torch.float32)
                        lowdim[0, 0] = this_lowdim

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

        print(f"Success rate: {np.sum(success) / env.num_init_states}")

    # -- rendering --
    elif mode == "rendering":
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
        obs, all_done, all_rew = env.reset(), False, 0

        image = torch.tensor(obs["agentview_image"] / 255.0, device=device, dtype=torch.float32)
        image = center_crop(image)
        image = image[None, None].repeat(1, To, 1, 1, 1)
        lowdim = np.concatenate(
            [obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]
        )
        lowdim = torch.tensor(lowdim, device=device, dtype=torch.float32)[None, None]

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
                condition_cfg={"image": image, "language": env.task_description, "lowdim": lowdim},
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

                    this_lowdim = np.concatenate(
                        [obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]
                    )
                    this_lowdim = torch.tensor(this_lowdim, device=device, dtype=torch.float32)
                    lowdim[0, 0] = this_lowdim

                if np.all(all_done):
                    break

            print("Rewards:", all_rew)

        writer = imageio.get_writer(default_root_dir / "video.mp4", fps=30)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        env.close()


# x0 = batch['x0']
# t = torch.rand((x0.shape[0],), device=x0.device)
# eps = torch.randn_like(x0)

# alpha, sigma = policy.noise_scheduler.t_to_schedule(t)
# alpha, sigma = at_least_ndim(alpha, x0.dim()), at_least_ndim(sigma, x0.dim())

# xt = alpha * x0 + sigma * eps
# xt = (1.0 - policy.fix_mask) * xt + policy.fix_mask * x0
# x = xt.clone()

# with torch.no_grad():
#     self = nn_diffusion
#     vec_condition = condition.get("vec_condition", None)
#     lang_condition = condition.get("lang_condition", None)
#     lang_condition_mask = condition.get("lang_condition_mask", None)
#     vis_condition = condition.get("vis_condition", None)
#     vis_condition_mask = condition.get("vis_condition_mask", None)

#     x = self.x_proj(x) + self.pos_emb
#     emb = self.t_proj(self.map_noise(t))

#     if condition is not None and vec_condition is not None:
#         emb = emb + vec_condition

#     x = nn_diffusion.blocks[0](x, emb, lang_condition, lang_condition_mask)
#     # x = nn_diffusion.blocks[1](x, emb, lang_condition, lang_condition_mask)
#     # x = nn_diffusion.blocks[2](x, emb, lang_condition, lang_condition_mask)

#     i = 1
#     block = nn_diffusion.blocks[i]
#     self = block
#     seq_condition = lang_condition if i % 2 == 0 else vis_condition
#     seq_condition_mask = lang_condition_mask if i % 2 == 0 else vis_condition_mask
#     vec_condition = emb

#     adaLN_coeff = self.adaLN_modulation(vec_condition.unsqueeze(-2))
#     if self._adaLN_on_cross_attn:
#         (
#             shift_sa,
#             scale_sa,
#             gate_sa,
#             shift_ca,
#             scale_ca,
#             gate_ca,
#             shift_ffn,
#             scale_ffn,
#             gate_ffn,
#         ) = adaLN_coeff.chunk(9, dim=-1)
#     else:
#         shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = adaLN_coeff.chunk(
#             6, dim=-1
#         )
#     if self.prenorm:
#         h = self.sa_norm(x) * (1 + scale_sa) + shift_sa
#         x = x + gate_sa * self.sa_attn(h, h, h)[0]
#         if self._adaLN_on_cross_attn:
#             h = self.ca_norm(x) * (1 + scale_ca) + shift_ca
#             add_on = gate_ca * self.ca_attn(
#                     h, seq_condition, seq_condition, key_padding_mask=seq_condition_mask
#             )[0]
#         else:
#             h = self.ca_norm(x)
#             add_on = self.ca_attn(
#                 h, seq_condition, seq_condition, key_padding_mask=seq_condition_mask
#             )[0]
#     else:
#         x = x + gate_sa * self.sa_attn(x, x, x)[0]
#         x = self.sa_norm(x) * (1 + scale_sa) + shift_sa
#         if self._adaLN_on_cross_attn:
#             add_on = gate_ca * self.ca_attn(
#                     x, seq_condition, seq_condition, key_padding_mask=seq_condition_mask
#             )[0]
#         else:
#             add_on = self.ca_attn(
#                 x, seq_condition, seq_condition, key_padding_mask=seq_condition_mask
#             )[0]

#     x_plt = torch.cat([x, add_on, x+add_on], 1)
#     plt.imshow(x_plt[0, :, ::4].cpu())
