"""Figures from the summary tables written by analyze.py (no simulation is run here).

fig1_main_onlyX_A        publication figure: cold-start accuracy across the three design slices
fig2_training_dynamics   diagnostic: per-block relative KL and posterior log-variance vs. iteration
fig3_mc_vs_plugin        method comparison: paired MC - plug-in difference in relative KL
fig4_heatmap_n_p         full n x p factorial at alpha = -6, relative KL on a diverging scale
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from src import config  # noqa: E402

G = config.GRID
BLOCK = config.MAIN_BLOCK

# Palette: ordinal blue ramp for ordered levels, fixed categorical order for blocks,
# diverging blue - gray - red around the baseline value 1.
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
CATEG = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "^", "s", "D", "v"]
INK, INK2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
BLOCK_NAMES = {"onlyX_A": "onlyX × A (cold start, main)", "onlyX_onlyA": "onlyX × onlyA",
               "onlyX_both": "onlyX × both", "A_A": "A × A (network observed)"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titlecolor": INK, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "lines.linewidth": 1.6, "lines.markersize": 4.5,
})


def style(ax, logy=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID_C, linewidth=0.6)
    ax.set_axisbelow(True)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def baseline(ax, y=1.0, text=None):
    ax.axhline(y, color=INK2, linestyle="--", linewidth=0.9, zorder=1)
    if text:
        ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=INK2)


def save(fig, name):
    os.makedirs(config.FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(config.FIG_DIR, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# the three slices of the design: (x variable, line variable, fixed condition, title)
SLICES = [
    ("n", "p", {"alpha": G["fixed_alpha"]}, f"vary n, p   (α = {G['fixed_alpha']})"),
    ("p", "alpha", {"n": G["fixed_n"]}, f"vary p, α   (n = {G['fixed_n']})"),
    ("n", "alpha", {"p": G["fixed_p"]}, f"vary n, α   (p = {G['fixed_p']})"),
]
LEVELS = {"n": G["n_levels"], "p": G["p_levels"], "alpha": G["alpha_levels"]}
SYM = {"n": "n", "p": "p", "alpha": "α"}


def slice_lines(ax, df, xvar, lvar, fixed, ycol="mean", lo="ci_lo", hi="ci_hi"):
    d = df
    for k, v in fixed.items():
        d = d[d[k] == v]
    for i, lv in enumerate(LEVELS[lvar]):
        g = d[d[lvar] == lv].sort_values(xvar)
        if g.empty:
            continue
        yerr = np.vstack([g[ycol] - g[lo], g[hi] - g[ycol]])
        ax.errorbar(g[xvar], g[ycol], yerr=yerr, color=ORDINAL[i], marker=MARKERS[i],
                    capsize=2, elinewidth=1, label=f"{SYM[lvar]} = {lv}", zorder=3)
    ax.set_xticks(LEVELS[xvar])
    ax.set_xlabel(SYM[xvar])


def fig1(summary):
    s = summary[(summary.block == BLOCK) & (summary.estimator == "mc")]
    rows = [("relkl", "relative KL\n(dashed 1 = constant-density predictor)", True,
             "constant predictor"),
            ("relfro_all", "relative Frobenius error\n(dashed 1 = all-zero predictor)", True,
             "all-zero predictor")]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharey="row")
    for r, (metric, ylab, logy, btxt) in enumerate(rows):
        d = s[s.metric == metric]
        for c, (xv, lv, fixed, title) in enumerate(SLICES):
            ax = axes[r, c]
            slice_lines(ax, d, xv, lv, fixed)
            style(ax, logy)
            baseline(ax, 1.0)
            if r == 0:
                ax.set_title(title)
                ax.legend(fontsize=7.5, loc="upper left" if c == 1 else "upper right",
                          handlelength=1.4)
            if c == 0:
                ax.set_ylabel(ylab)
    fig.suptitle("Cold-start edge probabilities (onlyX × A) are mostly worse than a constant predictor\n"
                 "MC posterior-mean estimator; mean over 20 replicates ± 95% Monte Carlo CI; lower is better",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig1_main_onlyX_A")


def fig2(traj):
    cases = [("N700_X100_Aneg6", "n=700, p=100, α=−6"), ("N700_X700_Aneg6", "n=700, p=700, α=−6"),
             ("N100_X1500_Aneg6", "n=100, p=1500, α=−6"), ("N700_X700_A0", "n=700, p=700, α=0")]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    for c, (lbl, title) in enumerate(cases):
        t = traj[traj.label == lbl]
        ax = axes[0, c]
        for i, b in enumerate(config.BLOCKS):
            g = t[t.variable == f"pi_relkl_{b}"].sort_values("iter")
            ax.plot(g["iter"], g["mean"], color=CATEG[i], marker=MARKERS[i], markevery=4,
                    label=BLOCK_NAMES[b], lw=2.2 if b == BLOCK else 1.4)
            ax.fill_between(g["iter"], g["mean"] - 1.96 * g["mcse"], g["mean"] + 1.96 * g["mcse"],
                            color=CATEG[i], alpha=0.18, lw=0)
        style(ax, logy=True)
        baseline(ax, 1.0)
        ax.set_title(title)
        if c == 0:
            ax.set_ylabel("relative KL, plug-in (log)\n(dashed 1 = constant predictor)")
        ax2 = axes[1, c]
        for i, (v, name) in enumerate([("logvar_mean_A", "network-observed nodes"),
                                       ("logvar_mean_onlyX", "onlyX nodes")]):
            g = t[t.variable == v].sort_values("iter")
            ax2.plot(g["iter"], g["mean"], color=CATEG[i], marker=MARKERS[i], markevery=4, label=name)
        style(ax2)
        ax2.set_xlabel("training iteration")
        if c == 0:
            ax2.set_ylabel("mean posterior\nlog-variance")
    axes[0, 0].legend(fontsize=7.5, loc="upper right")
    axes[1, 0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Diagnostic: the cold-start block is best after ~50–100 iterations, then degrades while A × A keeps improving;\n"
                 "posterior variances collapse early (mean ± 95% MC CI over 20 replicates)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig2_training_dynamics")


def fig3(paired):
    p = paired[(paired.metric == "relkl")].copy()
    p = p.rename(columns={"mean_diff": "mean"})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0), sharey=True)
    for c, (xv, lv, fixed, title) in enumerate(SLICES):
        ax = axes[c]
        slice_lines(ax, p, xv, lv, fixed)
        style(ax)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        baseline(ax, 0.0)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
                  handlelength=1.4)
        if c == 0:
            ax.set_ylabel("relative KL: MC − plug-in\n(< 0: MC better; dashed = 0; symlog)")
    fig.suptitle("Paired estimator comparison on onlyX × A: MC helps most where the plug-in is worst; "
                 "plug-in wins at small p / sparse graphs\n"
                 "mean paired difference over 20 replicates ± 95% Monte Carlo CI",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "fig3_mc_vs_plugin")


def fig4(summary):
    s = summary[(summary.block == BLOCK) & (summary.estimator == "mc") & (summary.metric == "relkl")
                & (summary.alpha == G["fixed_alpha"])]
    M = s.pivot(index="n", columns="p", values="mean").reindex(index=G["n_levels"], columns=G["p_levels"])
    L = np.log10(M.values)
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    lim = np.nanmax(np.abs(L))
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(L, cmap=cmap, norm=TwoSlopeNorm(0.0, -lim, lim), aspect="auto", origin="lower")
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(L[i, j]) > 0.6 * lim else INK)
    ax.set_xticks(range(len(G["p_levels"])), G["p_levels"])
    ax.set_yticks(range(len(G["n_levels"])), G["n_levels"])
    ax.set_xlabel("feature dimension p")
    ax.set_ylabel("number of nodes n")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    ticks = [v for v in [0.5, 1, 2, 5, 10, 20] if abs(np.log10(v)) <= lim]
    cb.set_ticks(np.log10(ticks), labels=[f"{v:g}" for v in ticks])
    cb.set_label("relative KL (1 = constant predictor)")
    ax.set_title(f"Cold-start relative KL, MC estimator, α = {G['fixed_alpha']}\n"
                 "red = worse than the constant predictor", fontsize=9.5, loc="left")
    fig.tight_layout()
    save(fig, "fig4_heatmap_n_p")


def main():
    R = config.RESULTS_DIR
    summary = pd.read_csv(os.path.join(R, "summary.csv"))
    traj = pd.read_csv(os.path.join(R, "trajectories.csv"))
    paired = pd.read_csv(os.path.join(R, "paired_mc_vs_pi.csv"))
    fig1(summary)
    fig2(traj)
    fig3(paired)
    fig4(summary)


if __name__ == "__main__":
    main()
