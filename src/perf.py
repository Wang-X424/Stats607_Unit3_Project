"""Component-level timing of one replicate, for the baseline (Unit 2) and optimised code.

Used by scripts/complexity.py and scripts/benchmark.py. Every timing is the median of
`repeats` wall-clock measurements after one warm-up call (execution time is a random variable).
"""
import time

import numpy as np
import torch

from . import config
from . import methods as M_opt
from . import metrics as E_opt
from .baseline import methods as M_base
from .baseline import metrics as E_base
from .dgps import generate_data, observe, split_nodes


def _median_time(fn, repeats):
    fn()  # warm-up (lazy imports, BLAS initialisation, allocator)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def setup_replicate(n, p, alpha, seed=42):
    Z, A, X = generate_data(n, p, alpha, seed=seed)
    obs = observe(Z, A, X, *split_nodes(n, seed=seed + 100))
    return obs


def time_components(n, p, alpha=-6, version="optimized", n_train_iter=20, repeats=3, seed=42):
    """Seconds for: dgp, preprocess, one training iteration, one evaluation call, MC estimator."""
    M = M_opt if version == "optimized" else M_base
    E = E_opt if version == "optimized" else E_base
    torch.manual_seed(seed)
    out = {"n": n, "p": p, "alpha": alpha, "version": version}

    out["dgp"] = _median_time(lambda: setup_replicate(n, p, alpha, seed), repeats)
    obs = setup_replicate(n, p, alpha, seed)
    ids = (obs["id_onlyA"], obs["id_both"], obs["id_onlyX"], obs["id_A"])
    out["preprocess"] = _median_time(
        lambda: M.preprocess(obs["X_obs"], obs["A_obs"], *ids, p, 1.0), repeats)

    # training: time n_train_iter iterations (setup included, amortised), report per iteration
    cfg = dict(config.MODEL, n_iter=n_train_iter)
    out["train_iter"] = _median_time(lambda: M.fit_vae(obs, p, cfg), repeats) / n_train_iter

    model, z_mu, z_lv = M.fit_vae(obs, p, dict(config.MODEL, n_iter=2))
    ng = torch.no_grad()      # as in the pipeline: evaluation never builds an autograd graph
    ng.__enter__()
    n_A = len(obs["id_A"])
    Zp = torch.tensor(obs["Z_ord"], dtype=torch.float64)
    blocks = E.make_blocks(obs)
    S = config.EVAL["mc_samples"]
    if version == "optimized":
        cols = torch.arange(n_A)
        theta = alpha + Zp @ Zp[:n_A].t()
        truth = E.prepare_truth(theta, blocks)
        out["eval"] = _median_time(lambda: E.evaluate_fast(model, z_mu, z_lv, truth, cols, False), repeats)
        out["mc"] = _median_time(lambda: M.mc_log_probs(z_mu, z_lv, model.PA_dens, S, cols), repeats)
        out["truth_setup"] = _median_time(lambda: E.prepare_truth(theta, blocks), repeats)
    else:
        theta = alpha + Zp @ Zp.t()
        out["eval"] = _median_time(lambda: E.evaluate(model, z_mu, z_lv, theta, blocks, False), repeats)
        out["mc"] = _median_time(lambda: M.mc_log_probs(z_mu, z_lv, model.PA_dens, S), repeats)
        out["truth_setup"] = 0.0
    ng.__exit__(None, None, None)
    return out


def replicate_seconds(c, n_iter=None, n_logs=None):
    """Projected wall time of one full replicate from component times."""
    n_iter = n_iter or config.MODEL["n_iter"]
    n_logs = n_logs or (n_iter // config.EVAL["log_every"] + 1)
    return (c["dgp"] + c["preprocess"] + c["truth_setup"] + n_iter * c["train_iter"]
            + n_logs * c["eval"] + c["mc"])


def _child(args):
    import resource
    out = time_components(*args[:3], **args[3])
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return out


def time_components_isolated(n, p, alpha=-6, **kw):
    """time_components in a fresh process: isolates memory (peak RSS is reported) and
    prevents allocator growth from one measurement leaking into the next."""
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:  # fork + OpenMP can deadlock
        return pool.map(_child, [(n, p, alpha, kw)])[0]
