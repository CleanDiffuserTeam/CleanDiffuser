import pytest
import torch
from cleandiffuser.nn_diffusion.jannerunet import JannerUNet1d

def test_jannerunet1d_forward():
    batch_size = 2
    horizon = 16
    in_dim = 16
    x = torch.randn(batch_size, horizon, in_dim)
    noise = torch.randn(batch_size)
    condition = torch.randn(batch_size, 32)

    model = JannerUNet1d(in_dim=in_dim, model_dim=32, emb_dim=32, kernel_size=3, dim_mult=[1, 2, 2, 2], attention=True)

    output = model(x, noise, condition)

    assert output.shape == (batch_size, horizon, in_dim), f"Expected shape {(batch_size, horizon, in_dim)}, but got {output.shape}"

def test_jannerunet1d_no_condition():
    batch_size = 2
    horizon = 8
    in_dim = 16
    x = torch.randn(batch_size, horizon, in_dim)
    noise = torch.randn(batch_size)

    model = JannerUNet1d(in_dim=in_dim, model_dim=32, emb_dim=32, kernel_size=3, dim_mult=[1, 2, 2, 2], attention=True)

    output = model(x, noise, None)

    assert output.shape == (batch_size, horizon, in_dim), f"Expected shape {(batch_size, horizon, in_dim)}, but got {output.shape}"
