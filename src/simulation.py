"""Simulation orchestration: scenario grid, replicate loop, caching of raw results.

Raw output format (unchanged from the original server run): one pickle per scenario,
``sens_<label>.pkl``, holding a list over replicates of a list over logged iterations of
dicts {'iter', 'loss', 'logvar_mean_onlyX', 'logvar_mean_A', '<est>_<metric>_<block>', ...}.
A tab-separated ``.txt`` copy is written next to each pickle.
"""
import logging
import os
import pickle
import time

import numpy as np
import torch

from . import config
from .dgps import generate_data, observe, split_nodes
from .methods import fit_vae
from .metrics import evaluate_fast, make_blocks, prepare_truth

log = logging.getLogger("simulation")

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    def tqdm(x, **kw):
        return x


def make_label(n, p, alpha):
    return f"N{n}_X{p}_A{str(alpha).replace('.', '').replace('-', 'neg')}"


def build_scenarios(grid=config.GRID):
    """Union of three two-way slices, de-duplicated by (n, p, alpha), in a fixed order."""
    seen = {}

    def add(n, p, a, slice_name):
        lbl = make_label(n, p, a)
        if lbl not in seen:
            seen[lbl] = {"label": lbl, "N_NODES": n, "X_DIM": p, "Alpha": a, "slices": []}
        seen[lbl]["slices"].append(slice_name)

    for n in grid["n_levels"]:
        for p in grid["p_levels"]:
            add(n, p, grid["fixed_alpha"], "n_x_p")
    for p in grid["p_levels"]:
        for a in grid["alpha_levels"]:
            add(grid["fixed_n"], p, a, "p_x_alpha")
    for n in grid["n_levels"]:
        for a in grid["alpha_levels"]:
            add(n, grid["fixed_p"], a, "n_x_alpha")
    return list(seen.values())


def log_iterations(n_iter, every):
    return np.insert(np.arange(every - 1, n_iter, every), 0, 0)


def run_one_rep(cfg, rep, grid=config.GRID, model_cfg=config.MODEL, eval_cfg=config.EVAL,
                device="cpu"):
    """One replicate: generate -> split/mask -> fit -> evaluate at logged iterations."""
    n, p, alpha = cfg["N_NODES"], cfg["X_DIM"], cfg["Alpha"]
    seed = grid["seed"] + rep

    Z, A, X = generate_data(n, p, alpha, seed=seed)
    idx_oA, idx_b, idx_oX = split_nodes(n, seed=seed + config.SPLIT["seed_offset"])
    obs = observe(Z, A, X, idx_oA, idx_b, idx_oX)

    n_A = len(obs["id_A"])
    Zp = torch.tensor(obs["Z_ord"], dtype=torch.float64, device=device)
    cols_needed = torch.arange(n_A, device=device)       # every block's columns lie in id_A
    theta_true = alpha + Zp @ Zp[:n_A].t()               # n x n_A instead of n x n
    blocks = make_blocks(obs, device)
    truth = prepare_truth(theta_true, blocks, eval_cfg["min_exp_deg"])

    iters = log_iterations(model_cfg["n_iter"], eval_cfg["log_every"])
    last = int(iters[-1])
    records = []

    def on_log(it, loss, model, zm, zlv):
        rc = {"iter": it, "loss": loss,
              "logvar_mean_onlyX": zlv[n_A:].mean().item(),
              "logvar_mean_A": zlv[:n_A].mean().item()}
        rc.update(evaluate_fast(model, zm, zlv, truth, cols_needed,
                                do_mc=(eval_cfg["mc_every_log"] or it == last),
                                mc_samples=eval_cfg["mc_samples"]))
        records.append(rc)

    # Seed torch per replicate (model init, reparameterisation noise, MC draws).
    # NOTE: the original server run did not seed torch; see README "Reproducibility".
    torch.manual_seed(seed)
    fit_vae(obs, p, model_cfg, device, log_iters=iters, on_log=on_log)
    return records


def run_one_scenario(cfg, grid=config.GRID, device="cpu", **kw):  # sequential path
    reps = []
    for rep in tqdm(range(grid["n_reps"]), desc=cfg["label"], leave=False):
        recs = run_one_rep(cfg, rep, grid=grid, device=device, **kw)
        f = recs[-1]
        log.info("%s rep %d/%d | onlyX_A relKL mc=%.4f pi=%.4f | A_A relKL pi=%.4f",
                 cfg["label"], rep + 1, grid["n_reps"], f["mc_relkl_onlyX_A"],
                 f["pi_relkl_onlyX_A"], f["pi_relkl_A_A"])
        reps.append(recs)
    return reps


def write_scenario_txt(res, path):
    cols = [k for k in res[0][-1].keys() if k not in ("iter", "loss")]
    with open(path, "w") as f:
        f.write("\t".join(["rep", "iter", "loss"] + cols) + "\n")
        for r, rep in enumerate(res):
            for rc in rep:
                row = [r, rc["iter"], rc["loss"]] + [rc.get(c, float("nan")) for c in cols]
                f.write("\t".join(str(v) for v in row) + "\n")


def _rep_job(cfg, rep, torch_threads, kw):
    """One (scenario, replicate) job for a worker process."""
    torch.set_num_threads(torch_threads)
    return cfg["label"], rep, run_one_rep(cfg, rep, **kw)


def run_all(out_dir=config.RAW_DIR, grid=config.GRID, device=None, force=False,
            n_jobs=None, torch_threads=None, scenarios=None, **kw):
    """Run every scenario not already cached in out_dir; returns {label: results}.

    n_jobs = 1 runs sequentially in this process (Unit 2 behaviour). Otherwise every
    (scenario, replicate) pair is an independent joblib job (process backend), so the work is
    parallel across replicates AND scenarios. Results do not depend on n_jobs: each replicate
    seeds its own RNGs (data: 42 + r, torch: 42 + r), and a scenario's pickle is written as
    soon as its last replicate finishes (resume-safe).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_jobs = config.PARALLEL["n_jobs"] if n_jobs is None else n_jobs
    torch_threads = torch_threads or config.PARALLEL["torch_threads"]
    os.makedirs(out_dir, exist_ok=True)
    scenarios = scenarios or build_scenarios(grid)
    kw = dict(kw, grid=grid, device=device)

    results, todo = {}, []
    for cfg in scenarios:
        path = os.path.join(out_dir, f"sens_{cfg['label']}.pkl")
        if os.path.exists(path) and not force:
            with open(path, "rb") as f:
                results[cfg["label"]] = pickle.load(f)
            log.info("[cached] %s", cfg["label"])
        else:
            todo.append(cfg)
    log.info("device=%s, %d/%d scenarios to run x %d reps, n_jobs=%s, output -> %s",
             device, len(todo), len(scenarios), grid["n_reps"], n_jobs, out_dir)

    def save(cfg, reps):
        path = os.path.join(out_dir, f"sens_{cfg['label']}.pkl")
        with open(path, "wb") as f:
            pickle.dump(reps, f)
        write_scenario_txt(reps, os.path.join(out_dir, f"sens_{cfg['label']}.txt"))
        results[cfg["label"]] = reps
        log.info("[saved] %s", cfg["label"])

    t0 = time.time()
    if n_jobs == 1:
        for cfg in tqdm(todo, desc="scenarios"):
            save(cfg, run_one_scenario(cfg, **kw))
    else:
        from joblib import Parallel, delayed
        by_label = {c["label"]: c for c in todo}
        pending = {c["label"]: [None] * grid["n_reps"] for c in todo}
        jobs = (delayed(_rep_job)(cfg, rep, torch_threads, kw)
                for cfg in todo for rep in range(grid["n_reps"]))
        stream = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(jobs)
        for lbl, rep, recs in tqdm(stream, total=len(todo) * grid["n_reps"], desc="replicates"):
            pending[lbl][rep] = recs
            if all(r is not None for r in pending[lbl]):
                save(by_label[lbl], pending.pop(lbl))
    log.info("finished %d scenarios in %.1fs", len(todo), time.time() - t0)
    return results
