import pytest
import torch
from cleandiffuser.utils import SUPPORTED_TIMESTEP_EMBEDDING
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.nn_diffusion import ChiTransformer

# base_nn_diffusion.py
@pytest.fixture
def base_nn_diffusion():
    emb_dim = 64
    timestep_emb_type = "positional"
    return BaseNNDiffusion(emb_dim, timestep_emb_type)

def test_initialization(base_nn_diffusion):
    assert isinstance(base_nn_diffusion, BaseNNDiffusion)

def test_forward_invalid_timestep_emb_type():
    with pytest.raises(AssertionError):
        invalid_timestep_emb_type = "invalid_type"
        BaseNNDiffusion(64, invalid_timestep_emb_type)


# chitransformer.py
@pytest.fixture
def model():
    return ChiTransformer(
        act_dim=10, obs_dim=5, Ta=15, To=5,
        d_model=256, nhead=4, num_layers=8,
        p_drop_emb=0.0, p_drop_attn=0.3,
        n_cond_layers=2, timestep_emb_type="positional"
    )

def test_chi_transformer_output_shape(model):
    batch_size = 4
    Ta = 15
    To = 5
    act_dim = 10
    obs_dim = 5

    x = torch.randn(batch_size, Ta, act_dim)
    noise = torch.randn(batch_size)
    condition = torch.randn(batch_size, To, obs_dim)

    output = model(x, noise, condition)

    assert output.shape == (batch_size, Ta, act_dim), "Output shape mismatch"

def test_chi_transformer_no_condition(model):
    batch_size = 4
    Ta = 15
    act_dim = 10

    x = torch.randn(batch_size, Ta, act_dim)
    noise = torch.randn(batch_size)

    with pytest.raises(TypeError):
        model(x, noise)

def test_chi_transformer_forward_pass(model):
    batch_size = 4
    Ta = 15
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
        ChiTransformer(
            act_dim=10, obs_dim=5, Ta=15, To=5,
            d_model=256, nhead=4, num_layers=8,
            p_drop_emb=0.0, p_drop_attn=0.3,
            n_cond_layers=2, timestep_emb_type="invalid_type"
        )

def test_chi_transformer_init_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            assert param.data.std().item() < 0.1, f"Weight init for {name} is not correct"
        if "bias" in name:
            assert param.data.mean().item() == 0, f"Bias init for {name} is not correct"



if __name__ == "__main__":
    pytest.main()
