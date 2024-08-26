import pytest
import torch
from torch.nn import MSELoss

from cleandiffuser.nn_classifier import HalfDiT1d, HalfJannerUNet1d, MLPNNClassifier, QGPONNClassifier, BaseNNClassifier

@pytest.fixture
def half_dit1d():
    return HalfDiT1d(in_dim=10, out_dim=1, emb_dim=64)


@pytest.fixture
def half_janner_unet1d():
    return HalfJannerUNet1d(horizon=32, in_dim=10, out_dim=1, emb_dim=64)


@pytest.fixture
def mlp_nn_classifier():
    return MLPNNClassifier(x_dim=10, out_dim=1, emb_dim=64, hidden_dims=[128, 64])


@pytest.fixture
def qgpo_nn_classifier():
    return QGPONNClassifier(obs_dim=10, act_dim=10, emb_dim=64, hidden_dims=[128, 64])


def test_half_dit1d_forward(half_dit1d):
    x = torch.randn(2, 32, 10)
    t = torch.randint(1000, (2,))
    condition = torch.randn(2, 64)

    output1 = half_dit1d(x, t)
    output2 = half_dit1d(x, t, condition)

    assert output1.shape == (2, 1)
    assert output2.shape == (2, 1)


def test_half_janner_unet1d_forward(half_janner_unet1d):
    x = torch.randn(2, 32, 10)
    t = torch.randint(1000, (2,))
    condition = torch.randn(2, 64)

    output1 = half_janner_unet1d(x, t)
    output2 = half_janner_unet1d(x, t, condition)

    assert output1.shape == (2, 1)
    assert output2.shape == (2, 1)


def test_mlp_nn_classifier_forward(mlp_nn_classifier):
    x = torch.randn(2, 10)
    t = torch.randint(1000, (2,))

    output = mlp_nn_classifier(x, t)

    assert output.shape == (2, 1)


def test_qgpo_nn_classifier_forward(qgpo_nn_classifier):
    x = torch.randn(2, 10)
    t = torch.randint(1000, (2,))
    y = torch.randn(2, 10)

    output = qgpo_nn_classifier(x, t, y)

    assert output.shape == (2, 1)
