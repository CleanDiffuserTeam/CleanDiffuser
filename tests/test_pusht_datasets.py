import os
import sys
import pytest
import torch
from cleandiffuser.dataset.pusht_dataset import PushTImageDataset
from cleandiffuser.dataset.pusht_dataset import PushTKeypointDataset
from cleandiffuser.dataset.pusht_dataset import PushTStateDataset


@pytest.fixture(scope="module")
def setup_environment():
    zarr_path = os.path.expanduser('dev/pusht/pusht_cchi_v7_replay.zarr')
    return zarr_path


@pytest.mark.parametrize("DatasetClass, expected_keys", [
    (PushTImageDataset, ["obs", "action"]),
    (PushTKeypointDataset, ["obs", "action"]),
    (PushTStateDataset, ["obs", "action"]),
])
def test_dataset_loading(setup_environment, DatasetClass, expected_keys):
    zarr_path = setup_environment
    dataset = DatasetClass(zarr_path, horizon=16, pad_before=0, pad_after=0)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    batch = next(iter(dataloader))

    assert all(
        key in batch for key in expected_keys), f"Batch keys {batch.keys()} do not match expected keys {expected_keys}"
    if DatasetClass == PushTImageDataset:
        assert "image" in batch["obs"], "Expected 'image' in batch['obs']"
        print("batch['obs']['image'].shape:", batch['obs']['image'].shape)
    if DatasetClass == PushTKeypointDataset:
        assert "keypoint" in batch["obs"], "Expected 'keypoint' in batch['obs']"
        print("batch['obs']['keypoint'].shape:", batch['obs']['keypoint'].shape)
    if DatasetClass == PushTStateDataset:
        assert "state" in batch["obs"], "Expected 'state' in batch['obs']"
        print("batch['obs']['state'].shape:", batch['obs']['state'].shape)

