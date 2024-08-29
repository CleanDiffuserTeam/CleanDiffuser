import pytest
import torch
import torch.nn as nn
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import GroupNorm1d
from cleandiffuser.nn_diffusion.pearcemlp import TimeSiren, FCBlock, PearceMlp


def test_pearce_mlp_forward():
    act_dim = 10
    To = 1
    emb_dim = 128
    hidden_dim = 512

    # Initialize PearceMlp instance
    model = PearceMlp(
        act_dim=act_dim,
        To=To,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim
    )

    # Create input tensors
    batch_size = 4
    x = torch.randn(batch_size, act_dim)
    noise = torch.randn(batch_size)
    condition = torch.randn(batch_size, To, emb_dim)

    # Perform forward pass
    output = model(x, noise, condition)

    # Check output tensor shape
    assert output.shape == (batch_size, act_dim), f"Expected output shape (b, {act_dim}), but got {output.shape}"

    # Check output tensor type
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor"

def test_pearce_mlp_forward_without_condition():
    act_dim = 10
    To = 1
    emb_dim = 128
    hidden_dim = 512

    # Initialize PearceMlp instance
    model = PearceMlp(
        act_dim=act_dim,
        To=To,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim
    )

    # Create input tensors
    batch_size = 4
    x = torch.randn(batch_size, act_dim)
    noise = torch.randn(batch_size)

    # Perform forward pass (without condition)
    output = model(x, noise)

    # Check output tensor shape
    assert output.shape == (batch_size, act_dim), f"Expected output shape (b, {act_dim}), but got {output.shape}"

    # Check output tensor type
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor"

def test_pearce_mlp_forward_with_different_hidden_dim():
    act_dim = 10
    To = 1
    emb_dim = 128
    hidden_dim = 256

    # Initialize PearceMlp instance
    model = PearceMlp(
        act_dim=act_dim,
        To=To,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim
    )

    # Create input tensors
    batch_size = 4
    x = torch.randn(batch_size, act_dim)
    noise = torch.randn(batch_size)
    condition = torch.randn(batch_size, To, emb_dim)

    # Perform forward pass
    output = model(x, noise, condition)

    # Check output tensor shape
    assert output.shape == (batch_size, act_dim), f"Expected output shape (b, {act_dim}), but got {output.shape}"

    # Check output tensor type
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor"