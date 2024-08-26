import pytest
import torch
import torch.nn as nn
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.nn_diffusion.pearcetransformer import TimeSiren, TransformerEncoderBlock, EmbeddingBlock, PearceTransformer

@pytest.fixture
def random_tensor():
    return torch.randn(2, 10)  # Example tensor for testing

def test_time_siren(random_tensor):
    model = TimeSiren(input_dim=10, emb_dim=20)
    output = model(random_tensor)
    assert output.shape == (2, 20)

def test_transformer_encoder_block():
    model = TransformerEncoderBlock(trans_emb_dim=16, transformer_dim=32, nheads=2)
    input_tensor = torch.randn(3, 2, 16)
    output = model(input_tensor)
    assert output.shape == (3, 2, 16)

def test_embedding_block(random_tensor):
    model = EmbeddingBlock(in_dim=10, emb_dim=20)
    output = model(random_tensor)
    assert output.shape == (2, 20)

def test_pearce_transformer():
    model = PearceTransformer(act_dim=10, To=2, emb_dim=32, trans_emb_dim=16, nhead=2)
    x = torch.randn(2, 10)
    noise = torch.randn(2)
    condition = torch.randn(2, 2, 32)
    output = model(x, noise, condition)
    assert output.shape == (2, 10)
