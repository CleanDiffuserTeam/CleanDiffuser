import pytest
import torch
import torch.nn as nn
from cleandiffuser.nn_diffusion.sfbc_unet import ResidualBlock, SfBCUNet


@pytest.fixture
def sample_data():
    x = torch.randn(2, 2, 10)
    noise = torch.randn(2, )
    condition = torch.randn(2, 64)  # Batch size of 2, condition size of 64
    return x, noise, condition

def test_residual_block_forward(sample_data):
    x, _, condition = sample_data
    in_dim, out_dim, emb_dim = 10, 10, 64
    block = ResidualBlock(in_dim, out_dim, emb_dim)
    output = block(x, condition)
    assert output.shape == x.shape

def test_sfbcu_net_forward(sample_data):
    x, noise, condition = sample_data
    act_dim = 10
    emb_dim = 64
    hidden_dims = [512, 256, 128]
    model = SfBCUNet(act_dim, emb_dim, hidden_dims)
    output = model(x, noise, condition)
    assert output.shape == x.shape

def test_sfbcu_net_forward_without_condition(sample_data):
    x, noise, _ = sample_data
    act_dim = 10
    emb_dim = 64
    hidden_dims = [512, 256, 128]
    model = SfBCUNet(act_dim, emb_dim, hidden_dims)
    output = model(x, noise)
    assert output.shape == x.shape