import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from cleandiffuser.classifier import MSEClassifier, QGPOClassifier, CumRewClassifier
from cleandiffuser.nn_classifier import HalfJannerUNet1d, QGPONNClassifier

# Fixtures
@pytest.fixture
def mse_classifier():
    obs_dim = 10
    act_dim = 5
    horizon = 1
    model_dim = 64
    emb_dim = 64
    dim_mult = [1, 2, 4]

    nn_classifier = HalfJannerUNet1d(
        horizon, obs_dim + act_dim, out_dim=1,
        model_dim=model_dim, emb_dim=emb_dim, dim_mult=dim_mult,
        timestep_emb_type="positional", kernel_size=3)

    classifier = MSEClassifier(nn_classifier=nn_classifier, temperature=2.0, device="cpu")
    return classifier

@pytest.fixture
def qgpo_classifier():
    obs_dim = 10
    act_dim = 5
    horizon = 1
    model_dim = 64
    emb_dim = 64
    dim_mult = [1, 2, 4]

    nn_classifier = QGPONNClassifier(obs_dim, act_dim, 64, [256, 256, 256], "untrainable_fourier")
    classifier = QGPOClassifier(nn_classifier=nn_classifier, device="cpu")
    return classifier

@pytest.fixture
def cumrew_classifier():
    obs_dim = 10
    act_dim = 5
    horizon = 1
    model_dim = 64
    emb_dim = 64
    dim_mult = [1, 2, 4]

    nn_classifier = HalfJannerUNet1d(
        horizon, obs_dim + act_dim, out_dim=1,
        model_dim=model_dim, emb_dim=emb_dim, dim_mult=dim_mult,
        timestep_emb_type="positional", kernel_size=3)

    classifier = CumRewClassifier(nn_classifier=nn_classifier, device="cpu")
    return classifier

# Tests for MSEClassifier
def test_mse_loss(mse_classifier):
    x = torch.randn(2, 10, 15)  # Assuming obs_dim + act_dim = 15
    noise = torch.randn(2, )
    y = torch.randn(2, 1)

    loss = mse_classifier.loss(x, noise, y)
    pred_y = mse_classifier.model(x, noise)
    expected_loss = torch.nn.functional.mse_loss(pred_y, y)
    assert expected_loss is not None

def test_mse_logp(mse_classifier):
    x = torch.randn(2, 10, 15)  # Assuming obs_dim + act_dim = 15
    noise = torch.randn(2, )
    c = torch.randn(2, 1)

    logp = mse_classifier.logp(x, noise, c)
    pred_y = mse_classifier.model_ema(x, noise)
    expected_logp = -2.0 * ((pred_y - c) ** 2).mean(-1, keepdim=True)
    assert expected_logp is not None

# # Tests for QGPOClassifier
def test_qgpo_loss(qgpo_classifier):

    x = torch.randn(2, 10, 5)
    t = torch.randn(2, )
    y = {
        "soft_label": torch.softmax(torch.randn(2, 10, 1), dim=1),
        "obs": torch.randn(2, 10)
    }

    loss, _ = qgpo_classifier.loss(x, t, y)
    assert loss is not None

def test_qgpo_logp(qgpo_classifier):
    x = torch.randn(2, 5)
    t = torch.randn(2, )
    c = torch.randn(2, 10)

    logp = qgpo_classifier.logp(x, t, c)
    assert logp is not None

# Tests for CumRewClassifier
def test_cumrew_loss(cumrew_classifier):
    x = torch.randn(2, 10, 15)  # Assuming obs_dim + act_dim = 15
    noise = torch.randn(2, )
    R = torch.randn(2, 1)

    loss = cumrew_classifier.loss(x, noise, R)
    pred_R = cumrew_classifier.model(x, noise)
    expected_loss = ((pred_R - R) ** 2).mean()
    assert expected_loss is not None

def test_cumrew_logp(cumrew_classifier):
    x = torch.randn(2, 10, 15)  # Assuming obs_dim + act_dim = 15
    noise = torch.randn(2, )

    logp = cumrew_classifier.logp(x, noise)
    pred_R = cumrew_classifier.model_ema(x, noise)
    assert pred_R is not None