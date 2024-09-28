import pytest
import torch
import torch.nn as nn
from cleandiffuser.nn_diffusion.idqlmlp import IDQLMlp, ResidualBlock, BaseNNDiffusion


def test_residual_block_forward():
    block = ResidualBlock(hidden_dim=64, dropout=0.1)
    x = torch.randn(2, 64)
    output = block(x)
    assert output.shape == x.shape

def test_idqlmlp_initialization():
    model = IDQLMlp(obs_dim=10, act_dim=5, emb_dim=64, hidden_dim=256, n_blocks=3, dropout=0.1)
    assert isinstance(model, BaseNNDiffusion)
    assert isinstance(model.time_mlp, nn.Sequential)
    assert isinstance(model.affine_in, nn.Linear)
    assert isinstance(model.ln_resnet, nn.Sequential)
    assert isinstance(model.affine_out, nn.Linear)

def test_idqlmlp_forward():
    model = IDQLMlp(obs_dim=10, act_dim=5, emb_dim=64, hidden_dim=256, n_blocks=3, dropout=0.1)
    x = torch.randn(2, 5)
    noise = torch.randn(2)
    condition = torch.randn(2, 10)
    output = model(x, noise, condition)
    assert output.shape == (2, 5)

def test_idqlmlp_forward_no_condition():
    model = IDQLMlp(obs_dim=10, act_dim=5, emb_dim=64, hidden_dim=256, n_blocks=3, dropout=0.1)
    x = torch.randn(2, 5)
    noise = torch.randn(2)
    output = model(x, noise)
    assert output.shape == (2, 5)

def test_idqlmlp_layer_shapes():
    model = IDQLMlp(obs_dim=10, act_dim=5, emb_dim=64, hidden_dim=256, n_blocks=3, dropout=0.1)
    x = torch.randn(2, 5)
    noise = torch.randn(2)
    condition = torch.randn(2, 10)
    t = model.time_mlp(model.map_noise(noise))
    assert t.shape == (2, 64)
    x_cat = torch.cat([x, t, condition], -1)
    assert x_cat.shape == (2, 79)
    affine_in_output = model.affine_in(x_cat)
    assert affine_in_output.shape == (2, 256)
