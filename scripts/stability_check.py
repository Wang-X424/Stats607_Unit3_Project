"""Numerical-stability audit across the simulation conditions.

    python scripts/stability_check.py [--live-reps 2]

1. Cached Unit 2 results (61 scenarios x 20 reps x 25 logged iterations):
   non-finite values per metric, where they occur, loss convergence over the last 100 iterations,
   posterior log-variance range.
2. DGP headroom (3 replicates per scenario): extreme float32 logits of A and X versus the
   float32 exp-overflow threshold, isolated network nodes, onlyA nodes with no 'both' neighbour
   (feature fill falls back to the global mean), rows excluded from relfro_avg.
3. Live training with every Python/NumPy/torch warning turned into an error, on the most extreme
   conditions (smallest n, largest p, sparsest and densest alpha): loss finite at every
   iteration, |log-variance| vs the lv_bound safeguard.

Writes results/perf/stability_report.md and results/perf/stability_*.csv.
"""
import argparse
import glob
import os
import pickle
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import config  # noqa: E402
from src.dgps import generate_data, observe, split_nodes  # noqa: E402
from src.methods import fit_vae  # noqa: E402
from src.simulation import build_scenarios  # noqa: E402

PERF = os.path.join(config.RESULTS_DIR, "perf")
F32_EXP_MAX = float(np.log(np.finfo(np.float32).max))  # ~88.72


def scan_cached(raw_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "sens_*.pkl"))):
        lbl = os.path.basename(path)[5:-4]
        with open(path, "rb") as f:
            reps = pickle.load(f)
        for r, rep in enumerate(reps):
            last, prev = rep[-1], rep[-3]           # iterations 1199 and 1099
            nonfinite = sorted(k for rc in rep for k, v in rc.items()
                               if k.startswith(("pi_", "mc_", "loss", "logvar"))
                               and not (k.startswith("mc_") and rc["iter"] != last["iter"])
                               and not np.isfinite(v))
            rows.append({"label": lbl, "rep": r,
                         "n_nonfinite": len(nonfinite), "nonfinite_keys": ";".join(sorted(set(nonfinite))),
                         "loss_final": last["loss"],
                         "loss_relchange_last100": (prev["loss"] - last["loss"]) / abs(last["loss"]),
                         "logvar_min": min(rc[k] for rc in rep for k in ("logvar_mean_A", "logvar_mean_onlyX")),
                         "logvar_max": max(rc[k] for rc in rep for k in ("logvar_mean_A", "logvar_mean_onlyX")),
                         "frac_rows_onlyX_A": last["pi_frac_rows_onlyX_A"]})
    return pd.DataFrame(rows)


def dgp_headroom(n_reps=config.GRID["n_reps"], n_graph_reps=3):
    rows = []
    d = config.DGP
    for c in build_scenarios():
        n, p, a = c["N_NODES"], c["X_DIM"], c["Alpha"]
        for r in range(n_reps):
            seed = config.GRID["seed"] + r
            rng = np.random.RandomState(seed)
            Z = rng.randn(n, d["z_dim_true"]).astype(np.float32) * d["z_scale"]
            logits_A = a + Z @ Z.T
            rng2 = np.random.RandomState(seed + d["seed_offset_x"])
            W1 = rng2.randn(d["z_dim_true"], d["x_hidden"]).astype(np.float32)
            b1 = rng2.randn(d["x_hidden"]).astype(np.float32) * 0.1
            W2 = rng2.randn(d["x_hidden"], p).astype(np.float32)
            b2 = rng2.randn(p).astype(np.float32) * 0.1
            logits_X = np.maximum(0, Z @ W1 + b1) @ W2 + b2
            row = {"label": c["label"], "n": n, "p": p, "alpha": a, "rep": r,
                   "min_logit_A": float(logits_A.min()), "max_logit_A": float(logits_A.max()),
                   "min_logit_X": float(logits_X.min()), "max_logit_X": float(logits_X.max()),
                   "frac_isolated_obs": np.nan, "frac_onlyA_no_both_nb": np.nan, "obs_edges": np.nan}
            if r < n_graph_reps:   # graph-structure checks need the full replicate
                Zg, A, X = generate_data(n, p, a, seed=seed)
                obs = observe(Zg, A, X, *split_nodes(n, seed=seed + 100))
                A_obs = obs["A_obs"]
                row.update(frac_isolated_obs=float((A_obs.sum(1) == 0).mean()),
                           frac_onlyA_no_both_nb=float(
                               (A_obs[np.ix_(obs["id_onlyA"], obs["id_both"])].sum(1) == 0).mean()),
                           obs_edges=int(A_obs.sum() // 2))
            rows.append(row)
    df = pd.DataFrame(rows)
    df["x_headroom"] = F32_EXP_MAX - (-df[["min_logit_X", "min_logit_A"]].min(axis=1))
    return df


def live(reps):
    conds = [(100, 1500, -8), (100, 1500, 0), (100, 100, -8), (100, 100, 0), (400, 1500, -8)]
    rows = []
    for n, p, a in conds:
        for r in range(reps):
            seed = config.GRID["seed"] + r
            Z, A, X = generate_data(n, p, a, seed=seed)
            obs = observe(Z, A, X, *split_nodes(n, seed=seed + 100))
            losses, lvmax = [], [0.0]

            def on_log(it, loss, model, zm, zlv):
                losses.append(loss)
                lvmax[0] = max(lvmax[0], float(zlv.abs().max()))

            iters = range(config.MODEL["n_iter"])          # log every iteration
            torch.manual_seed(seed)
            err = ""
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                old = np.seterr(over="raise", invalid="raise", divide="raise")
                try:
                    fit_vae(obs, p, config.MODEL, log_iters=iters, on_log=on_log)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                finally:
                    np.seterr(**old)
            rows.append({"n": n, "p": p, "alpha": a, "rep": r, "error": err,
                         "all_losses_finite": bool(np.all(np.isfinite(losses))),
                         "max_abs_logvar": lvmax[0], "lv_bound": config.MODEL["lv_bound"]})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-reps", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(PERF, exist_ok=True)

    cached = scan_cached(config.RAW_DIR)
    head = dgp_headroom()
    lv = live(a.live_reps)
    cached.to_csv(os.path.join(PERF, "stability_cached.csv"), index=False)
    head.to_csv(os.path.join(PERF, "stability_dgp.csv"), index=False)
    lv.to_csv(os.path.join(PERF, "stability_live.csv"), index=False)

    nf = cached[cached.n_nonfinite > 0]
    keys = sorted({k for s in nf.nonfinite_keys for k in s.split(";") if k})
    rep = ["# Stability report", "",
           f"## Cached Unit 2 results ({cached.label.nunique()} scenarios, {len(cached)} replicates)", "",
           f"* replicates with any non-finite logged value: **{len(nf)}** "
           f"({len(nf) / len(cached):.1%}); scenarios: {', '.join(sorted(nf.label.unique())) or 'none'}",
           f"* non-finite keys: {', '.join(keys) or 'none'}",
           f"* final loss finite in all replicates: {bool(np.isfinite(cached.loss_final).all())}",
           f"* relative loss decrease over the last 100 iterations: median "
           f"{cached.loss_relchange_last100.median():.2e}, max {cached.loss_relchange_last100.max():.2e}, "
           f"replicates where the loss went UP: {(cached.loss_relchange_last100 < 0).sum()}",
           f"* mean posterior log-variance range over all logs: "
           f"[{cached.logvar_min.min():.2f}, {cached.logvar_max.max():.2f}]",
           "", "## DGP headroom (20 replicates x 61 scenarios; graph checks on 3)", "",
           f"* replicates where the Unit 2 naive sigmoid overflowed float32 exp (RuntimeWarning): "
           f"**{(head.min_logit_X < -F32_EXP_MAX).sum()} / {len(head)}** "
           f"({(head.min_logit_X < -F32_EXP_MAX).mean():.1%}) in "
           f"{head[head.min_logit_X < -F32_EXP_MAX].label.nunique()} scenarios",
           f"* most negative float32 logit: A {head.min_logit_A.min():.1f}, X {head.min_logit_X.min():.1f} "
           f"(exp overflows float32 beyond {F32_EXP_MAX:.1f}); smallest headroom {head.x_headroom.min():.1f}",
           f"* largest logit: A {head.max_logit_A.max():.1f}, X {head.max_logit_X.max():.1f}",
           f"* observed graphs with zero edges: {(head.obs_edges == 0).sum()}",
           f"* isolated observed nodes: max fraction {head.frac_isolated_obs.max():.2f} "
           f"(at {head.loc[head.frac_isolated_obs.idxmax(), 'label']})",
           f"* onlyA nodes with no 'both' neighbour (global-mean fill): max fraction "
           f"{head.frac_onlyA_no_both_nb.max():.2f} (at {head.loc[head.frac_onlyA_no_both_nb.idxmax(), 'label']})",
           "", "## Live training with warnings as errors", "",
           "```", lv.to_string(index=False), "```"]
    with open(os.path.join(PERF, "stability_report.md"), "w") as f:
        f.write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
