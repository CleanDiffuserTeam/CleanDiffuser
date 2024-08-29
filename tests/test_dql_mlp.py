import pytest
import torch
from cleandiffuser.nn_diffusion import BaseNNDiffusion, DQLMlp


def test_dqlmlp_forward():
    model = DQLMlp(obs_dim=10, act_dim=5, emb_dim=16)
    x = torch.randn(2, 5)
    noise = torch.randn(2)
    condition = torch.randn(2, 10)

    output = model(x, noise, condition)
    assert output.shape == (2, 5)


def test_dqlmlp_forward_no_condition():
    model = DQLMlp(obs_dim=10, act_dim=5, emb_dim=16)
    x = torch.randn(2, 5)  # batch size of 2, act_dim of 5
    noise = torch.randn(2)  # batch size of 2

    with pytest.raises(TypeError):
        model(x, noise)  # condition is required, should raise TypeError


def test_dqlmlp_forward_with_different_shapes():
    model = DQLMlp(obs_dim=10, act_dim=5, emb_dim=16)
    x = torch.randn(3, 5)  # batch size of 3, act_dim of 5
    noise = torch.randn(3)  # batch size of 3
    condition = torch.randn(3, 10)  # batch size of 3, obs_dim of 10

    output = model(x, noise, condition)
    assert output.shape == (3, 5)  # should match (batch_size, act_dim)


def test_dqlmlp_layer_shapes():
    model = DQLMlp(obs_dim=10, act_dim=5, emb_dim=16)
    x = torch.randn(2, 5)  # batch size of 2, act_dim of 5
    noise = torch.randn(2)  # batch size of 2
    condition = torch.randn(2, 10)  # batch size of 2, obs_dim of 10

    t = model.time_mlp(model.map_noise(noise))
    assert t.shape == (2, 16)

    x_cat = torch.cat([x, t, condition], -1)
    assert x_cat.shape == (2, 31)

    mid_output = model.mid_layer(x_cat)
    assert mid_output.shape == (2, 256)