"""Simulation-budget extension: how many replicates does each scenario actually need?

Uses only the cached Unit 2 results (20 replicates per scenario) as a pilot study and the
per-replicate cost of the optimised code (results/perf/benchmark_pairs.csv).

Target: Monte Carlo SE of the primary metric (MC relative KL, onlyX x A) at most `--rel`
(default 2%) of its mean, i.e. a 95% CI half-width of about 4% of the value. Required replicates
R(s) = ceil((sd_upper / (rel * mean))^2), with sd_upper the 95% upper confidence bound for the SD
from the pilot (chi-square, R-1 df), floored at --min-reps.

    python scripts/budget.py [--rel 0.02] [--min-reps 5]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src import config  # noqa: E402

PERF = os.path.join(config.RESULTS_DIR, "perf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel", type=float, default=0.02)
    ap.add_argument("--min-reps", type=int, default=5)
    a = ap.parse_args()

    s = pd.read_csv(os.path.join(config.RESULTS_DIR, "summary.csv"))
    m = s[(s.block == "onlyX_A") & (s.estimator == "mc") & (s.metric == "relkl")].copy()
    R = m.n_reps
    m["sd_upper"] = m.sd * np.sqrt((R - 1) / stats.chi2.ppf(0.05, R - 1))
    m["reps_needed"] = np.maximum(a.min_reps, np.ceil((m.sd_upper / (a.rel * m["mean"])) ** 2)).astype(int)

    cost = pd.read_csv(os.path.join(PERF, "benchmark_pairs.csv"))
    cost = cost[cost.version == "optimized"][["n", "p", "replicate_s"]]
    m = m.merge(cost, on=["n", "p"], how="left")
    m["cost_equal_h"] = m.replicate_s * config.GRID["n_reps"] / 3600
    m["cost_adaptive_h"] = m.replicate_s * m.reps_needed / 3600
    m[["label", "n", "p", "alpha", "mean", "sd", "sd_upper", "reps_needed", "replicate_s",
       "cost_equal_h", "cost_adaptive_h"]].to_csv(os.path.join(PERF, "budget.csv"), index=False)

    by_n = m.groupby("n").agg(sd=("sd", "median"), reps=("reps_needed", "median"),
                              cost_s=("replicate_s", "median")).reset_index()
    out = {"target_rel_mcse": a.rel, "min_reps": a.min_reps,
           "equal_allocation_hours": float(m.cost_equal_h.sum()),
           "adaptive_allocation_hours": float(m.cost_adaptive_h.sum()),
           "total_reps_equal": int(config.GRID["n_reps"] * len(m)),
           "total_reps_adaptive": int(m.reps_needed.sum()),
           "scenarios_needing_more_than_20": m[m.reps_needed > 20].label.tolist(),
           "median_by_n": by_n.round(3).to_dict(orient="records")}
    with open(os.path.join(PERF, "budget_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for n, g in m.groupby("n"):
        axes[0].scatter(g.replicate_s, g.reps_needed, s=18, label=f"n = {n}")
    axes[0].axhline(config.GRID["n_reps"], ls="--", color="#52514e", lw=0.9)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("seconds per replicate (optimised)")
    axes[0].set_ylabel(f"replicates for MCSE ≤ {a.rel:.0%} of mean")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("expensive scenarios need few replicates", fontsize=9.5, loc="left")
    tot = m.groupby("n")[["cost_equal_h", "cost_adaptive_h"]].sum()
    x = np.arange(len(tot))
    axes[1].bar(x - 0.2, tot.cost_equal_h, 0.4, color="#eb6834", label="20 reps everywhere")
    axes[1].bar(x + 0.2, tot.cost_adaptive_h, 0.4, color="#2a78d6", label="adaptive")
    axes[1].set_xticks(x, tot.index)
    axes[1].set_xlabel("n"); axes[1].set_ylabel("compute hours (this machine)")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].set_title("compute by n", fontsize=9.5, loc="left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(config.FIG_DIR, f"perf_budget.{ext}"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
