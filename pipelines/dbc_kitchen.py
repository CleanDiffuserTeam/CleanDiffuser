from cleandiffuser.dataset.kitchen_dataset import KitchenDataset
from cleandiffuser.nn_diffusion import PearceMlp, PearceTransformer
import pytorch_lightning as L
from pathlib import Path
import torch


class BC_Wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        batch = self.sampler.sample_sequence(idx)
        return {"x0": batch["action"], "condition_cfg": batch["state"]}


if __name__ == "__main__":
    seed = 0
    nn_backbone = "pearce_mlp"

    L.seed_everything(seed)

    save_path = Path(__file__).parents[1] / f"results/dbc_kitchen/{nn_backbone}/"

    # --- Create Dataset ---
    dataset = KitchenDataset("dev/kitchen", horizon=2, pad_before=1, pad_after=0)
    


    pass
