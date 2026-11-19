"""Regression tests: the optimised code (src/) reproduces the Unit 2 baseline (src/baseline/).

Three levels of evidence:
  1. deterministic components agree to floating-point precision (preprocess, encoder,
     evaluation metrics, MC estimator with the same draws);
  2. whole replicates with the same seed agree; differences start at float32 rounding
     (~1e-9 relative) because the optimised loop evaluates mathematically identical matrix
     products on different shapes, and stay far below the Monte Carlo error after 1,200 iterations;
  3. distributional: rerunning a cached scenario with the optimised code gives final metrics
     whose distribution matches the cached Unit 2 server results (which used unseeded torch,
     so only a distributional comparison is meaningful).
Parallel execution is checked to be bit-identical to sequential execution.
"""
import glob
import os
import pickle

import numpy as np
import pytest
import torch
from scipy import stats

from src import config
from src import methods as M
from src import metrics as E
from src.baseline import methods as Mb
from src.baseline import metrics as Eb
from src.baseline.simulation import run_one_rep as run_base
from src.dgps import generate_data, observe, split_nodes
from src.simulation import run_all, run_one_rep

from .conftest import TINY_EVAL, TINY_GRID, TINY_MODEL

KEYS = ["mc_relkl_onlyX_A", "pi_relkl_onlyX_A", "mc_relfro_all_onlyX_A", "pi_relkl_A_A"]


def _obs(n=150, p=80, alpha=-4, seed=3):
    Z, A, X = generate_data(n, p, alpha, seed=seed)
    return observe(Z, A, X, *split_nodes(n, seed=seed + 100)), p, alpha


# ---------------------------------------------------------------- level 1: components
@pytest.mark.parametrize("n,p,alpha", [(60, 20, -8), (150, 80, -4), (300, 50, 0)])
def test_preprocess_identical(n, p, alpha):
    obs, p, _ = _obs(n, p, alpha)
    ids = (obs["id_onlyA"], obs["id_both"], obs["id_onlyX"], obs["id_A"])
    a = Mb.preprocess(obs["X_obs"], obs["A_obs"], *ids, p, 1.0)
    b = M.preprocess(obs["X_obs"], obs["A_obs"], *ids, p, 1.0)
    assert a[0] == b[0]
    np.testing.assert_allclose(a[1], b[1], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(a[2], b[2], rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(a[3], b[3])


def test_encoder_with_precomputed_product_matches():
    obs, p, _ = _obs()
    ids = (obs["id_onlyA"], obs["id_both"], obs["id_onlyX"], obs["id_A"])
    _, X_fill, A_hat, _ = M.preprocess(obs["X_obs"], obs["A_obs"], *ids, p, 1.0)
    torch.manual_seed(0)
    model = M.VAE(p, 32, 8, -2.0)
    Xf = torch.tensor(X_fill, dtype=torch.float32)
    AH = torch.tensor(A_hat, dtype=torch.float32)
    n_A, n_oX = len(obs["id_A"]), len(obs["id_onlyX"])
    mu0, lv0 = model.Q(Xf, AH, n_A, n_oX)
    mu1, lv1 = model.Q_pre(AH @ Xf[:n_A], AH, Xf[n_A:], lv_bound=30.0)
    torch.testing.assert_close(mu0, mu1, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(lv0, lv1, rtol=1e-5, atol=1e-6)


def test_evaluation_identical_including_mc():
    obs, p, alpha = _obs()
    model, z_mu, z_lv = M.fit_vae(obs, p, dict(TINY_MODEL, n_iter=5))
    n_A = len(obs["id_A"])
    Zp = torch.tensor(obs["Z_ord"], dtype=torch.float64)
    blocks = E.make_blocks(obs)
    torch.manual_seed(7)
    a = Eb.evaluate(model, z_mu, z_lv, alpha + Zp @ Zp.t(), blocks, True, mc_samples=10)
    truth = E.prepare_truth(alpha + Zp @ Zp[:n_A].t(), blocks)
    torch.manual_seed(7)
    b = E.evaluate_fast(model, z_mu, z_lv, truth, torch.arange(n_A), True, mc_samples=10)
    assert a.keys() == b.keys()
    for k in a:
        if np.isnan(a[k]):
            assert np.isnan(b[k]), k
        else:
            assert b[k] == pytest.approx(a[k], rel=1e-10, abs=1e-13), k


def test_logvar_guard_is_inactive_on_observed_range():
    lv = torch.linspace(-15, 15, 301)   # observed max |lv| is about 12
    assert torch.equal(lv.clamp(-config.MODEL["lv_bound"], config.MODEL["lv_bound"]), lv)


# ---------------------------------------------------------------- level 2: same-seed replicates
CFG = {"label": "t", "N_NODES": 200, "X_DIM": 150, "Alpha": -4}


def test_short_run_matches_to_float32_rounding():
    mc, ev = dict(config.MODEL, n_iter=30), dict(config.EVAL, log_every=3)
    a = run_base(CFG, 0, model_cfg=mc, eval_cfg=ev)
    b = run_one_rep(CFG, 0, model_cfg=mc, eval_cfg=ev)
    assert [r["iter"] for r in a] == [r["iter"] for r in b]
    for ra, rb in zip(a, b):
        for k in ra:
            if np.isnan(ra[k]):
                assert np.isnan(rb[k])
            else:
                assert rb[k] == pytest.approx(ra[k], rel=1e-5, abs=1e-8), (ra["iter"], k)


@pytest.mark.parametrize("cfg", [{"label": "a", "N_NODES": 100, "X_DIM": 100, "Alpha": -6},
                                 {"label": "b", "N_NODES": 100, "X_DIM": 700, "Alpha": 0}])
def test_full_length_replicates_agree_far_below_mc_error(cfg):
    for rep in range(2):
        a, b = run_base(cfg, rep)[-1], run_one_rep(cfg, rep)[-1]
        for k in KEYS:
            assert b[k] == pytest.approx(a[k], rel=2e-2), (rep, k)


# ---------------------------------------------------------------- level 3: distributional
@pytest.mark.parametrize("label", ["N100_X100_Aneg6", "N100_X700_A0"])
def test_optimised_rerun_matches_cached_unit2_distribution(label, tmp_path):
    path = os.path.join(config.RAW_DIR, f"sens_{label}.pkl")
    if not os.path.exists(path):
        pytest.skip("cached Unit 2 results not available")
    with open(path, "rb") as f:
        cached = pickle.load(f)
    cfg = next(c for c in __import__("src.simulation", fromlist=["x"]).build_scenarios()
               if c["label"] == label)
    new = run_all(out_dir=str(tmp_path), scenarios=[cfg], n_jobs=-1)[label]
    for k in KEYS:
        x = np.array([r[-1][k] for r in cached])
        y = np.array([r[-1][k] for r in new])
        se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        assert abs(x.mean() - y.mean()) < 3 * se, (k, x.mean(), y.mean(), se)
        assert stats.ks_2samp(x, y).pvalue > 0.01, k


# ---------------------------------------------------------------- parallel == sequential
def test_parallel_equals_sequential(tmp_path):
    grid = dict(config.GRID, n_reps=3)
    scen = [{"label": "p1", "N_NODES": 60, "X_DIM": 30, "Alpha": -2},
            {"label": "p2", "N_NODES": 80, "X_DIM": 20, "Alpha": -6}]
    kw = dict(grid=grid, scenarios=scen, model_cfg=TINY_MODEL, eval_cfg=TINY_EVAL)
    seq = run_all(out_dir=str(tmp_path / "s"), n_jobs=1, **kw)
    par = run_all(out_dir=str(tmp_path / "p"), n_jobs=2, torch_threads=1, **kw)
    for lbl in seq:
        for ra, rb in zip(seq[lbl], par[lbl]):
            for ca, cb in zip(ra, rb):
                for k in ca:
                    assert (np.isnan(ca[k]) and np.isnan(cb[k])) or ca[k] == cb[k], (lbl, k)
    assert sorted(glob.glob(str(tmp_path / "p" / "*.pkl")))


# ---------------------------------------------------------------- DGP overflow fix
def test_stable_sigmoid_matches_naive_dgp():
    """The overflow-safe sigmoid generates bit-identical A and X, without the RuntimeWarning."""
    import warnings
    from src.baseline.dgps import generate_data as gen_base
    for n, p, a, s in [(700, 400, -6, 42), (200, 1500, -8, 3), (100, 100, 0, 7)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Zb, Ab, Xb = gen_base(n, p, a, seed=s)
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # optimised DGP must not warn
            Zo, Ao, Xo = generate_data(n, p, a, seed=s)
        np.testing.assert_array_equal(Zb, Zo)
        np.testing.assert_array_equal(Ab, Ao)
        np.testing.assert_array_equal(Xb, Xo)


def test_stable_sigmoid_values():
    from src.dgps import sigmoid
    x = np.array([-1000, -100, -88, -10, 0, 10, 100], dtype=np.float32)
    with np.errstate(over="raise"):
        y = sigmoid(x)
    assert np.all(np.isfinite(y)) and np.all((y >= 0) & (y <= 1))
    mid = np.array([-10, 0, 10], dtype=np.float32)
    np.testing.assert_array_equal(sigmoid(mid), 1.0 / (1.0 + np.exp(-mid)))
