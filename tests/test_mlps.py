import pytest
import torch
import torch.nn as nn
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import Mlp
from cleandiffuser.nn_diffusion.mlps import MlpNNDiffusion


def test_mlp_nn_diffusion_forward():
    x_dim = 10
    emb_dim = 16
    hidden_dims = [256, 256]
    activation = nn.ReLU()

    # Initialize MlpNNDiffusion instance
    model = MlpNNDiffusion(
        x_dim=x_dim,
        emb_dim=emb_dim,
        hidden_dims=hidden_dims,
        activation=activation
    )

    # Create input tensors
    batch_size = 4
    x = torch.randn(batch_size, x_dim)
    noise = torch.randn(batch_size)
    condition = torch.randn(batch_size, emb_dim)

    # Perform forward pass
    output = model(x, noise, condition)

    # Check output tensor shape
    assert output.shape == (batch_size, x_dim), f"Expected output shape (b, {x_dim}), but got {output.shape}"

    # Check output tensor type
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor"

def test_mlp_nn_diffusion_forward_without_condition():
    x_dim = 10
    emb_dim = 16
    hidden_dims = [256, 256]
    activation = nn.ReLU()

    # Initialize MlpNNDiffusion instance
    model = MlpNNDiffusion(
        x_dim=x_dim,
        emb_dim=emb_dim,
        hidden_dims=hidden_dims,
        activation=activation
    )

    # Create input tensors
    batch_size = 4
    x = torch.randn(batch_size, x_dim)
    noise = torch.randn(batch_size)

    # Perform forward pass (without condition)
    output = model(x, noise)

    # Check output tensor shape
    assert output.shape == (batch_size, x_dim), f"Expected output shape (b, {x_dim}), but got {output.shape}"

    # Check output tensor type
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor"

def test_mlp_nn_diffusion_forward_with_different_hidden_dims():
    x_dim = 10
    emb_dim = 16
    hidden_dims = [128, 128, 64]
    activation = nn.ReLU()

    # Initialize MlpNNDiffusion instance
    model = MlpNNDiffusion(
        x_dim=x_dim,
        emb_dim=emb_dim,
        hidden_dims=hidden_dims,
        activation=activation
    )

    # Create input tensors
    batch_size = 4
    x = torch.randn(batch_size, x_dim)
    noise = torch.randn(batch_size)
    condition = torch.randn(batch_size, emb_dim)

    # Perform forward pass
    output = model(x, noise, condition)

    # Check output tensor shape
    assert output.shape == (batch_size, x_dim), f"Expected output shape (b, {x_dim}), but got {output.shape}"

    # Check output tensor type
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor"