from pathlib import Path

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from cleandiffuser.dataset.kitchen_dataset import KitchenDataset
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_condition import PearceObsCondition
from cleandiffuser.nn_diffusion import PearceMlp, PearceTransformer


class BC_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        batch = self.sampler.sample_sequence(idx)
        return {"x0": batch["action"][-1], "condition_cfg": batch["state"]}


if __name__ == "__main__":
    seed = 0
    nn_backbone = "pearce_transformer"
    mode = "training"

    assert nn_backbone in ["pearce_mlp", "pearce_transformer"]

    L.seed_everything(seed)

    save_path = Path(__file__).parents[1] / f"results/dbc_kitchen/{nn_backbone}/"

    # --- Create Dataset ---
    # 3 observaitions = 2 history + 1 current observation
    # 1 action = 1 current action
    dataset = KitchenDataset(
        Path(__file__).parents[1] / "dev/kitchen", horizon=3, pad_before=2, pad_after=0, abs_action=False
    )
    obs_dim, act_dim = dataset.obs_dim, dataset.act_dim

    # --- Create Diffusion Model ---
    if nn_backbone == "pearce_mlp":
        nn_diffusion = PearceMlp(
            x_dim=act_dim, condition_horizon=3, emb_dim=128, timestep_emb_type="untrainable_fourier"
        )
    else:
        nn_diffusion = PearceTransformer(
            x_dim=act_dim, condition_horizon=3, emb_dim=128, timestep_emb_type="untrainable_fourier"
        )

    nn_condition = PearceObsCondition(obs_dim=obs_dim, emb_dim=128, flatten=False, dropout=0.0)

    actor = ContinuousDiffusionSDE(
        nn_diffusion,
        nn_condition,
        ema_rate=0.999,
        x_max=torch.full((act_dim,), 1.0),
        x_min=torch.full((act_dim,), -1.0),
    )

    # --- Training ---
    if mode == "training":
        dataloader = torch.utils.data.DataLoader(
            dataset=BC_Wrapper(dataset),
            batch_size=512,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

        callback = ModelCheckpoint(dirpath=save_path, filename="dbc-{step}", every_n_train_steps=100_000, save_top_k=-1)

        trainer = L.Trainer(
            accelerator="gpu",
            devices=[0, 1, 2, 3],
            max_steps=500_000,
            deterministic=True,
            log_every_n_steps=500,
            default_root_dir=save_path,
            callbacks=[callback],
        )

        trainer.fit(actor, dataloader)
