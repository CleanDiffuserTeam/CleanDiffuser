import sys
import os
import pytest
import torch
from cleandiffuser.dataset.kitchen_dataset import KitchenMjlDataset, KitchenDataset

@pytest.fixture
def setup_dataloader(request):
    abs_action = request.param
    if abs_action:
        dataset_path = os.path.expanduser("dev/kitchen/kitchen_demos_multitask")
        dataset = KitchenMjlDataset(dataset_path, horizon=10, pad_before=0, pad_after=0)
    else:
        dataset_path = os.path.expanduser("dev/kitchen")
        dataset = KitchenDataset(dataset_path, horizon=10, pad_before=0, pad_after=0)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )
    return dataloader

@pytest.mark.parametrize("setup_dataloader", [True, False], indirect=True)
def test_dataloader_batch(setup_dataloader):
    dataloader = setup_dataloader

    batch = next(iter(dataloader))

    assert 'obs' in batch
    assert 'action' in batch

    print("batch['obs']['state'].shape:", batch['obs']['state'].shape)
    print("batch['action'].shape:", batch['action'].shape)

    assert batch['obs']['state'].shape == (10, 10, 60)
    assert batch['action'].shape == (10, 10, 9)


if __name__ == '__main__':
    pytest.main()