"""Summarise cached raw results into tidy CSV tables (no simulation is run here).

Outputs (results/):
    design.csv            one row per scenario: n, p, alpha, slices, node counts
    final_reps.csv        one row per scenario x replicate, all metrics at the final iteration
    summary.csv           mean, SD, Monte Carlo SE and 95% CI per scenario x estimator x block x metric
    paired_mc_vs_pi.csv   paired MC - plug-in differences (main block) with MCSE
    trajectories.csv      mean / MCSE over replicates of plug-in metrics at each logged iteration
    key_numbers.json      headline numbers quoted in ANALYSIS.md
"""
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

from src import config
from src.simulation import build_scenarios

Z95 = 1.959964


def load_raw(raw_dir=config.RAW_DIR):
    out = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "sens_*.pkl"))):
        with open(path, "rb") as f:
            out[os.path.basename(path)[5:-4]] = pickle.load(f)
    if not out:
        raise SystemExit(f"No raw results in {raw_dir}; run `make simulate` first.")
    return out


def design_table(scen):
    rows = []
    for c in scen:
        n = c["N_NODES"]
        n_oA = int(np.round(n * config.SPLIT["onlyA"] / 100))
        n_b = int(np.round(n * config.SPLIT["both"] / 100))
        n_oX = min(int(np.round(n * config.SPLIT["onlyX"] / 100)), n - n_oA - n_b)
        rows.append({"label": c["label"], "n": n, "p": c["X_DIM"], "alpha": c["Alpha"],
                     "slices": "+".join(c["slices"]), "n_onlyA": n_oA, "n_both": n_b, "n_onlyX": n_oX})
    return pd.DataFrame(rows)


def final_reps(raw, design):
    rows = []
    for lbl, reps in raw.items():
        for r, rep in enumerate(reps):
            d = {"label": lbl, "rep": r}
            d.update({k: v for k, v in rep[-1].items()})
            rows.append(d)
    df = pd.DataFrame(rows)
    return design[["label", "n", "p", "alpha"]].merge(df, on="label")


def summary_table(fr):
    keys = [c for c in fr.columns if c.split("_")[0] in ("pi", "mc")]
    long = fr.melt(id_vars=["label", "n", "p", "alpha", "rep"], value_vars=keys,
                   var_name="key", value_name="value")
    parts = long["key"].str.extract(r"^(pi|mc)_(.+?)_(onlyX_A|onlyX_onlyA|onlyX_both|A_A)$")
    long["estimator"], long["metric"], long["block"] = parts[0], parts[1], parts[2]
    g = long.dropna(subset=["value"]).groupby(["label", "n", "p", "alpha", "estimator", "block", "metric"])
    s = g["value"].agg(mean="mean", sd="std", n_reps="count").reset_index()
    s["mcse"] = s["sd"] / np.sqrt(s["n_reps"])
    s["ci_lo"], s["ci_hi"] = s["mean"] - Z95 * s["mcse"], s["mean"] + Z95 * s["mcse"]
    return s


def paired_table(fr, block=config.MAIN_BLOCK):
    rows = []
    for (lbl, n, p, a), g in fr.groupby(["label", "n", "p", "alpha"]):
        for m in ["relkl", "relfro_all", "relfro_avg"]:
            d = g[f"mc_{m}_{block}"] - g[f"pi_{m}_{block}"]
            rows.append({"label": lbl, "n": n, "p": p, "alpha": a, "block": block, "metric": m,
                         "mean_diff": d.mean(), "mcse_diff": d.std(ddof=1) / np.sqrt(len(d)),
                         "frac_mc_better": float((d < 0).mean())})
    out = pd.DataFrame(rows)
    out["ci_lo"] = out["mean_diff"] - Z95 * out["mcse_diff"]
    out["ci_hi"] = out["mean_diff"] + Z95 * out["mcse_diff"]
    return out


def trajectory_table(raw, design):
    vars_ = (["loss", "logvar_mean_onlyX", "logvar_mean_A"]
             + [f"pi_{m}_{b}" for b in config.BLOCKS for m in ("relkl", "relfro_all", "relfro_theta")])
    rows = []
    for lbl, reps in raw.items():
        iters = [rc["iter"] for rc in reps[0]]
        for v in vars_:
            M = np.array([[rc[v] for rc in rep] for rep in reps], dtype=float)
            sd = M.std(0, ddof=1)
            for j, it in enumerate(iters):
                rows.append({"label": lbl, "iter": it, "variable": v, "mean": M[:, j].mean(),
                             "sd": sd[j], "mcse": sd[j] / np.sqrt(M.shape[0])})
    return design[["label", "n", "p", "alpha"]].merge(pd.DataFrame(rows), on="label")


def key_numbers(s, paired, traj, design):
    main = s[(s.block == config.MAIN_BLOCK)]
    mc_kl = main[(main.estimator == "mc") & (main.metric == "relkl")]
    mc_fro = main[(main.estimator == "mc") & (main.metric == "relfro_all")]
    aa_kl = s[(s.block == "A_A") & (s.estimator == "mc") & (s.metric == "relkl")]
    kn = {
        "n_scenarios": int(len(design)),
        "n_reps_main": int(mc_kl.n_reps.min()),
        "mc_relkl_main_gt1_ci": int((mc_kl.ci_lo > 1).sum()),
        "mc_relkl_main_lt1_ci": int((mc_kl.ci_hi < 1).sum()),
        "mc_relkl_main_lt1_scenarios": mc_kl[mc_kl.ci_hi < 1].label.tolist(),
        "mc_relfro_all_main_gt1": int((mc_fro["mean"] > 1).sum()),
        "mc_relkl_AA_lt1": int((aa_kl["mean"] < 1).sum()),
        "max_rel_mcse_main_relkl": float((mc_kl.mcse / mc_kl["mean"]).max()),
        "median_rel_mcse_main_relkl": float((mc_kl.mcse / mc_kl["mean"]).median()),
    }
    pk = paired[paired.metric == "relkl"]
    kn["paired_relkl_mc_better_ci"] = int((pk.ci_hi < 0).sum())
    kn["paired_relkl_pi_better_ci"] = int((pk.ci_lo > 0).sum())
    kn["paired_relkl_pi_better_scenarios"] = pk[pk.ci_lo > 0].label.tolist()
    kn["paired_relkl_max_abs_diff"] = float(pk.mean_diff.abs().max())

    # where along training is the main-block relKL minimised (plug-in, mean over reps)?
    t = traj[traj.variable == f"pi_relkl_{config.MAIN_BLOCK}"]
    best = t.loc[t.groupby("label")["mean"].idxmin()]
    final = t[t.iter == t.iter.max()].set_index("label")["mean"]
    kn["argmin_iter_main_relkl_counts"] = {int(k): int(v) for k, v in best.iter.value_counts().sort_index().items()}
    kn["median_ratio_final_over_best_main_relkl"] = float((final / best.set_index("label")["mean"]).median())
    lv = traj[(traj.iter == traj.iter.max()) & traj.variable.isin(["logvar_mean_onlyX", "logvar_mean_A"])]
    kn["final_logvar_range"] = [float(lv["mean"].min()), float(lv["mean"].max())]
    return kn


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    raw = load_raw()
    design = design_table(build_scenarios())
    missing = set(design.label) - set(raw)
    if missing:
        print(f"WARNING: {len(missing)} scenarios missing from raw results: {sorted(missing)[:5]} ...")
    design = design[design.label.isin(raw)]

    fr = final_reps(raw, design)
    s = summary_table(fr)
    paired = paired_table(fr)
    traj = trajectory_table(raw, design)
    kn = key_numbers(s, paired, traj, design)

    R = config.RESULTS_DIR
    design.to_csv(os.path.join(R, "design.csv"), index=False)
    fr.to_csv(os.path.join(R, "final_reps.csv"), index=False)
    s.to_csv(os.path.join(R, "summary.csv"), index=False)
    paired.to_csv(os.path.join(R, "paired_mc_vs_pi.csv"), index=False)
    traj.to_csv(os.path.join(R, "trajectories.csv"), index=False)
    with open(os.path.join(R, "key_numbers.json"), "w") as f:
        json.dump(kn, f, indent=2)
    print(json.dumps(kn, indent=2))
    print(f"Wrote summaries for {len(design)} scenarios to {R}/")


if __name__ == "__main__":
    main()
