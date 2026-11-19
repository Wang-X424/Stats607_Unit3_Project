"""Method/metric correctness on cases with known answers."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from src.dgps import generate_data
from src.methods import mc_log_probs, plugin_log_probs, preprocess
from src.metrics import bern_kl, prob_metrics


def _theta(n=80, alpha=-2.0, seed=0):
    Z, _, _ = generate_data(n, 5, alpha, seed=seed)
    Z = torch.tensor(Z, dtype=torch.float64)
    return alpha + Z @ Z.T


def test_perfect_estimate_has_zero_error():
    th = _theta()
    m = prob_metrics(th, F.logsigmoid(th), F.logsigmoid(-th))
    for k in ["relfro_all", "relfro_avg", "relfro_cent", "rmse", "kl", "relkl"]:
        assert abs(m[k]) < 1e-10, k


def test_constant_density_predictor_is_the_baseline():
    """P_hat = pbar gives relKL = 1 and relfro_cent = 1 by construction."""
    th = _theta()
    pbar = torch.sigmoid(th).mean()
    q = torch.full_like(th, float(pbar))
    m = prob_metrics(th, torch.log(q), torch.log1p(-q))
    assert abs(m["relkl"] - 1) < 1e-10
    assert abs(m["relfro_cent"] - 1) < 1e-10


def test_near_zero_predictor_has_relfro_all_one():
    th = _theta()
    q = torch.full_like(th, 1e-300)
    m = prob_metrics(th, torch.log(q), torch.log1p(-q))
    assert abs(m["relfro_all"] - 1) < 1e-10


def test_kl_nonnegative_and_mask_excludes_entries():
    torch.manual_seed(0)
    th = _theta()
    noisy = th + torch.randn_like(th)
    kl = bern_kl(F.logsigmoid(th), F.logsigmoid(-th), F.logsigmoid(noisy), F.logsigmoid(-noisy))
    assert (kl >= -1e-12).all()
    # corrupting only masked-out entries must not change the metrics
    mask = ~torch.eye(th.shape[0], dtype=torch.bool)
    bad = noisy.clone()
    bad.fill_diagonal_(50.0)
    a = prob_metrics(th, F.logsigmoid(noisy), F.logsigmoid(-noisy), mask)
    b = prob_metrics(th, F.logsigmoid(bad), F.logsigmoid(-bad), mask)
    assert all(math.isclose(a[k], b[k]) for k in a)


def test_mc_estimator_reduces_to_plugin_when_posterior_is_degenerate():
    torch.manual_seed(1)
    mu = torch.randn(30, 4)
    lv = torch.full_like(mu, -80.0)  # posterior sd ~ e^-40
    b = torch.tensor(-1.0)
    l1, l0 = mc_log_probs(mu, lv, b, S=7)
    p1, p0 = plugin_log_probs(mu, b)
    assert torch.allclose(l1, p1, atol=1e-10) and torch.allclose(l0, p0, atol=1e-10)


def test_mc_probabilities_are_valid_and_shrink_toward_half():
    """log P_hat and log(1-P_hat) must be consistent; averaging sigmoids tempers extremes (Jensen)."""
    torch.manual_seed(2)
    mu = 2 * torch.randn(40, 4)
    lv = torch.zeros_like(mu)
    b = torch.tensor(0.0)
    l1, l0 = mc_log_probs(mu, lv, b, S=200)
    assert torch.allclose(torch.exp(l1) + torch.exp(l0), torch.ones_like(l1), atol=1e-10)
    p1, _ = plugin_log_probs(mu, b)
    off = ~torch.eye(40, dtype=torch.bool)
    assert ((torch.exp(l1) - 0.5).abs()[off].mean() < (torch.exp(p1) - 0.5).abs()[off].mean())


def test_preprocess_normalised_adjacency_is_symmetric():
    _, A, X = generate_data(40, 10, -1, seed=3)
    ids = np.arange(40)
    id_oA, id_b, id_oX = ids[:12], ids[12:28], ids[28:]
    A_obs = A[:28, :28]
    X_obs = X[12:]
    lep, X_fill, A_hat, mask = preprocess(X_obs, A_obs, id_oA, id_b, id_oX, ids[:28], 10, 1.0)
    assert np.allclose(A_hat, A_hat.T)
    assert X_fill.shape == (40, 10) and np.all((X_fill >= 0) & (X_fill <= 1))
    assert mask.sum() == 28 * 27 // 2
