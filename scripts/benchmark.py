"""Baseline vs optimised timing comparison.

    python scripts/benchmark.py [--quick] [--max-jobs K]

1. Per-replicate time for every distinct (n, p) pair of the 61-scenario design (alpha does not
   affect cost), projected from component timings (src/perf.py), and the implied total runtime of
   the full study (61 scenarios x 20 replicates x 1,200 iterations), sequential.
2. A measured full-length replicate to validate the projection.
3. Parallel speedup: 2K replicates of one scenario with 1..K worker processes (1 torch thread each).

Writes results/perf/benchmark_*.csv, results/perf/benchmark_summary.json and
results/figures/perf_benchmark.{pdf,png}.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import config  # noqa: E402
from src.baseline.simulation import run_one_rep as run_base  # noqa: E402
from src.perf import replicate_seconds, time_components_isolated  # noqa: E402
from src.simulation import build_scenarios, run_all, run_one_rep  # noqa: E402

PERF = os.path.join(config.RESULTS_DIR, "perf")
COLORS = {"baseline": "#eb6834", "optimized": "#2a78d6"}


def per_pair(pairs, repeats, n_train_iter):
    rows = []
    for v in ["baseline", "optimized"]:
        for n, p in pairs:
            c = time_components_isolated(n, p, version=v, repeats=repeats, n_train_iter=n_train_iter)
            c["replicate_s"] = replicate_seconds(c)
            rows.append(c)
            print(f"{v:9s} n={n:4d} p={p:4d}  {c['replicate_s']:7.2f} s/replicate", flush=True)
    return pd.DataFrame(rows)


def validate(n=400, p=400):
    cfg = {"label": "v", "N_NODES": n, "X_DIM": p, "Alpha": -6}
    out = {}
    for v, run in [("baseline", run_base), ("optimized", run_one_rep)]:
        run(cfg, 1, model_cfg=dict(config.MODEL, n_iter=20))
        t0 = time.perf_counter()
        run(cfg, 0)
        out[v] = time.perf_counter() - t0
    return out


def parallel_speedup(max_jobs, n_iter):
    cfg = {"label": "par", "N_NODES": 400, "X_DIM": 400, "Alpha": -6}
    n_reps = 4 * max_jobs
    grid = dict(config.GRID, n_reps=n_reps)
    rows = []
    import tempfile
    for jobs, threads in [(1, None)] + [(k, 1) for k in range(1, max_jobs + 1)]:
        with tempfile.TemporaryDirectory() as d:
            if threads is None:  # sequential in-process, default torch threading (Unit 2 style)
                label = f"sequential ({torch.get_num_threads()} torch threads)"
            else:
                label = f"{jobs} worker(s) x 1 thread"
            default_threads = torch.get_num_threads()
            if threads == 1 and jobs == 1:   # sequential path runs in-process: pin it to 1 thread
                torch.set_num_threads(1)
            t0 = time.perf_counter()
            run_all(out_dir=d, grid=grid, scenarios=[cfg], n_jobs=1 if threads is None else jobs,
                    torch_threads=threads or torch.get_num_threads(),
                    model_cfg=dict(config.MODEL, n_iter=n_iter))
            wall = time.perf_counter() - t0
            torch.set_num_threads(default_threads)
            rows.append({"config": label, "jobs": jobs, "threads": threads or default_threads,
                         "wall_s": wall, "n_reps": n_reps})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    ref = df[(df.jobs == 1) & (df.threads == 1)].wall_s.iloc[0]
    df["speedup_vs_1worker"] = ref / df.wall_s
    return df


def plot(pp, study, par):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    w = pp.pivot_table(index=["n", "p"], columns="version", values="replicate_s").reset_index()
    w["speedup"] = w.baseline / w.optimized
    for v, m in [("baseline", "o"), ("optimized", "s")]:
        ax.scatter(w.n * w.p, w[v], color=COLORS[v], marker=m, s=22, label=v, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n × p (25 distinct cost cells of the design)")
    ax.set_ylabel("projected seconds per replicate")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"per replicate: speedup {w.speedup.min():.2f}–{w.speedup.max():.2f}×", fontsize=9.5, loc="left")

    ax = axes[1]
    labels = list(study.keys())
    vals = [study[k] / 3600 for k in labels]
    cols = [COLORS["baseline"]] + [COLORS["optimized"]] * (len(labels) - 1)
    ax.barh(labels[::-1], vals[::-1], color=cols[::-1], height=0.55)
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v:.1f} h", va="center", fontsize=8)
    ax.set_xlabel("projected wall time of the full study (hours, this 2-core machine)")
    ax.set_title("full study: 61 scenarios × 20 replicates", fontsize=9.5, loc="left")
    ax.set_xlim(0, max(vals) * 1.25)

    ax = axes[2]
    q = par[par.threads == 1]
    ax.plot(q.jobs, q.speedup_vs_1worker, marker="s", color=COLORS["optimized"], label="measured")
    ax.plot(q.jobs, q.jobs, ls="--", color="#52514e", lw=0.9, label="ideal (linear)")
    seq = par[par.threads != 1]
    if len(seq):
        ax.axhline(seq.speedup_vs_1worker.iloc[0], color=COLORS["baseline"], lw=0.9, ls=":",
                   label=seq.config.iloc[0])
    ax.set_xticks(q.jobs)
    ax.set_xlabel(f"worker processes (cores available: {os.cpu_count()})")
    ax.set_ylabel("speedup vs 1 worker")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("parallel replicates (n = p = 400)", fontsize=9.5, loc="left")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(config.FIG_DIR, f"perf_benchmark.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=os.cpu_count())
    ap.add_argument("--reuse-pairs", action="store_true",
                    help="reuse results/perf/benchmark_pairs.csv and only redo validation + parallel")
    a = ap.parse_args()
    os.makedirs(PERF, exist_ok=True)

    scen = build_scenarios()
    pairs = sorted({(c["N_NODES"], c["X_DIM"]) for c in scen})
    if a.quick:
        pairs = [(100, 100), (400, 400), (700, 700)]
    if a.reuse_pairs:
        pp = pd.read_csv(os.path.join(PERF, "benchmark_pairs.csv"))
    else:
        pp = per_pair(pairs, repeats=1, n_train_iter=8)
        pp.to_csv(os.path.join(PERF, "benchmark_pairs.csv"), index=False)

    cost = pp.set_index(["version", "n", "p"]).replicate_s
    reps = config.GRID["n_reps"]
    total = {v: sum(cost[(v, c["N_NODES"], c["X_DIM"])] for c in scen
                    if (v, c["N_NODES"], c["X_DIM"]) in cost.index) * reps
             for v in ["baseline", "optimized"]}

    val = validate()
    par = parallel_speedup(a.max_jobs, n_iter=200 if a.quick else 600)
    par.to_csv(os.path.join(PERF, "benchmark_parallel.csv"), index=False)
    best = par[par.threads == 1].speedup_vs_1worker.max()
    seq_ratio = (par[par.threads != 1].wall_s.iloc[0] / par[(par.jobs == 1) & (par.threads == 1)].wall_s.iloc[0])
    study = {"baseline, sequential": total["baseline"],
             "optimised, sequential": total["optimized"],
             # sequential timings above use default torch threading -> scale by the measured
             # wall-time ratio (K workers x 1 thread) / (sequential, default threads)
             f"optimised, {a.max_jobs} workers": total["optimized"]
             * par[par.threads == 1].wall_s.min() / par[par.threads != 1].wall_s.iloc[0]}
    summary = {"study_hours": {k: v / 3600 for k, v in study.items()},
               "measured_full_replicate_n400_p400_s": val,
               "per_replicate_speedup_range": [float((cost["baseline"] / cost["optimized"]).min()),
                                               float((cost["baseline"] / cost["optimized"]).max())],
               "overall_sequential_speedup": total["baseline"] / total["optimized"],
               "parallel_best_speedup": float(best), "cpu_count": os.cpu_count(),
               "sequential_2threads_over_1thread_walltime": float(seq_ratio),
               "note": "projected from component timings on this machine; scenarios with n = p = 1500 dominate"}
    with open(os.path.join(PERF, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    plot(pp, study, par)


if __name__ == "__main__":
    main()
