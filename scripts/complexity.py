"""Empirical computational complexity: component run time vs n (p = 700) and vs p (n = 700),
baseline vs optimised. Fits log-log slopes on the larger sizes.

    python scripts/complexity.py [--repeats 3] [--quick]

Writes results/perf/complexity.csv, results/perf/complexity_slopes.csv and
results/figures/perf_complexity.{pdf,png}.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import config  # noqa: E402
from src.perf import replicate_seconds, time_components_isolated as time_components  # noqa: E402

PERF = os.path.join(config.RESULTS_DIR, "perf")
COMP = ["train_iter", "eval", "mc", "preprocess", "dgp"]
COLORS = {"baseline": "#eb6834", "optimized": "#2a78d6"}


def run(n_grid, p_grid, repeats, n_train_iter):
    rows = []
    for v in ["baseline", "optimized"]:
        for n in n_grid:
            rows.append(dict(time_components(n, 700, version=v, repeats=repeats,
                                             n_train_iter=n_train_iter), sweep="n"))
            print(v, "n", n, f"{rows[-1]['train_iter'] * 1e3:.2f} ms/iter", flush=True)
        for p in p_grid:
            rows.append(dict(time_components(700, p, version=v, repeats=repeats,
                                             n_train_iter=n_train_iter), sweep="p"))
            print(v, "p", p, f"{rows[-1]['train_iter'] * 1e3:.2f} ms/iter", flush=True)
    df = pd.DataFrame(rows)
    df["replicate_s"] = df.apply(replicate_seconds, axis=1)
    return df


def slopes(df, min_size=400):
    out = []
    for (sweep, v), g in df.groupby(["sweep", "version"]):
        x = g["n" if sweep == "n" else "p"]
        g = g[x >= min_size]
        x = np.log(g["n" if sweep == "n" else "p"])
        for c in COMP + ["replicate_s", "peak_rss_mb"]:
            b, a = np.polyfit(x, np.log(g[c]), 1)
            out.append({"sweep": sweep, "version": v, "component": c, "slope": b})
    return pd.DataFrame(out)


def plot(df, sl):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [("n", "replicate_s", "projected time per replicate (s)"),
              ("n", "train_iter", "time per training iteration (s)"),
              ("p", "replicate_s", "projected time per replicate (s)")]
    for ax, (sweep, comp, ylab) in zip(axes, panels):
        g = df[df.sweep == sweep]
        xv = "n" if sweep == "n" else "p"
        for v in ["baseline", "optimized"]:
            h = g[g.version == v].sort_values(xv)
            s = sl[(sl.sweep == sweep) & (sl.version == v) & (sl.component == comp)].slope.iloc[0]
            ax.plot(h[xv], h[comp], marker="o" if v == "baseline" else "s", color=COLORS[v],
                    label=f"{v}  (slope {s:.2f})")
            big = h[h[xv] >= 400]
            b, a = np.polyfit(np.log(big[xv]), np.log(big[comp]), 1)
            xx = np.linspace(big[xv].min(), big[xv].max(), 20)
            ax.plot(xx, np.exp(a) * xx ** b, ls="--", lw=0.9, color=COLORS[v])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"{xv}  ({'p = 700' if xv == 'n' else 'n = 700'}, log scale)")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e4e3df", lw=0.6, which="both")
    fig.suptitle("Empirical complexity, baseline vs optimised (median of repeated timings; "
                 "dashed = log-log fit on sizes ≥ 400)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(config.FIG_DIR, f"perf_complexity.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(PERF, exist_ok=True)
    os.makedirs(config.FIG_DIR, exist_ok=True)
    print("torch threads:", torch.get_num_threads())
    if a.quick:
        df = run([100, 200, 400, 700], [100, 400, 700], 1, 5)
    else:
        df = run([100, 200, 400, 700, 1100, 1500], [100, 400, 700, 1100, 1500], a.repeats, 10)
    sl = slopes(df)
    df.to_csv(os.path.join(PERF, "complexity.csv"), index=False)
    sl.to_csv(os.path.join(PERF, "complexity_slopes.csv"), index=False)
    print(sl.pivot_table(index=["sweep", "component"], columns="version", values="slope").round(2))
    plot(df, sl)
