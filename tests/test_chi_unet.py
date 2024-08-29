import pytest
import torch
from cleandiffuser.nn_diffusion.chiunet import ChiUNet1d

@pytest.fixture
def model():
    return ChiUNet1d(
        act_dim=10, obs_dim=5, To=5,
        model_dim=256, emb_dim=256, kernel_size=5,
        cond_predict_scale=True, obs_as_global_cond=True,
        dim_mult=[1, 2, 2], timestep_emb_type="positional"
    )

def test_chiunet1d_output_shape(model):
    batch_size = 4
    Ta = 16
    To = 5
    act_dim = 10
    obs_dim = 5

    x = torch.randn(batch_size, Ta, act_dim)
    noise = torch.randn(batch_size)
    condition = torch.randn(batch_size, To, obs_dim)

    output = model(x, noise, condition)

    assert output.shape == (batch_size, Ta, act_dim), "Output shape mismatch"

def test_chiunet1d_no_condition(model):
    batch_size = 4
    Ta = 16
    act_dim = 10

    x = torch.randn(batch_size, Ta, act_dim)
    noise = torch.randn(batch_size)

    with pytest.raises(TypeError):
        model(x, noise)

def test_chiunet1d_forward_pass(model):
    batch_size = 4
    Ta = 16
    To = 5
    act_dim = 10
    obs_dim = 5

    x = torch.randn(batch_size, Ta, act_dim)
    noise = torch.randn(batch_size)
    condition = torch.randn(batch_size, To, obs_dim)

    try:
        output = model(x, noise, condition)
    except Exception as e:
        pytest.fail(f"Forward pass failed with exception: {e}")

def test_invalid_timestep_emb_type():
    with pytest.raises(AssertionError):
        ChiUNet1d(
            act_dim=10, obs_dim=5, To=5,
            model_dim=256, emb_dim=256, kernel_size=5,
            cond_predict_scale=True, obs_as_global_cond=True,
            dim_mult=[1, 2, 2], timestep_emb_type="invalid_type"
        )

def test_chiunet1d_ta_shape(model):
    batch_size = 4
    Ta = 14 # not 2^n dimension
    To = 5
    act_dim = 10
    obs_dim = 5

    x = torch.randn(batch_size, Ta, act_dim)
    noise = torch.randn(batch_size)
    condition = torch.randn(batch_size, To, obs_dim)

    with pytest.raises(AssertionError):
        output = model(x, noise, condition)
