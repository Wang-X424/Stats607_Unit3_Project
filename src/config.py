"""Single source of truth for every simulation parameter.

Nothing numeric is hard-coded elsewhere in src/; every function receives the
values it needs from the dictionaries below.  Setting the environment variable
SIM_QUICK=1 swaps in a tiny grid for smoke tests (``make simulate-quick``).
"""
import math
import os

# ------------------------------------------------------------------
# Data-generating process (held fixed across scenarios)
# ------------------------------------------------------------------
DGP = {
    "z_dim_true": 8,        # d, latent dimension of the truth
    "z_var": 1.0,           # Var(Z_ik)
    "x_hidden": 32,         # hidden width of the random ReLU generator for X
    "seed_offset_x": 1000,  # X generator weights/noise use seed + 1000
}
DGP["z_scale"] = math.sqrt(DGP["z_var"])

# Node-type split, in percent of n:  onlyA (network only) / both / onlyX (features only)
SPLIT = {"onlyA": 30, "both": 40, "onlyX": 30, "seed_offset": 100}

# ------------------------------------------------------------------
# Fitted model (VAE with GCN encoder and inner-product edge decoder)
# ------------------------------------------------------------------
MODEL = {
    "z_dim_fit": 8,
    "h_dim_fit": 128,
    "lr": 1e-3,
    "lr_min": 1e-5,          # cosine-annealing floor
    "l2": 1e-2,
    "self_weight": 1.0,      # self-loop weight (times degree) in A_tilde
    "tau2": DGP["z_var"],    # prior variance in the KL term, matched to the truth
    "n_iter": 1200,
    "lv_bound": 30.0,        # Unit 3 overflow guard on the posterior log-variance (never active in practice)
}

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
EVAL = {
    "mc_samples": 50,        # posterior draws for the MC estimator
    "mc_every_log": False,   # MC metrics only at the final logged iteration
    "min_exp_deg": 1.0,      # relfro_avg skips rows with expected degree < this
    "log_every": 50,         # metrics logged at iter 0, 49, 99, ...
}

# ------------------------------------------------------------------
# Scenario grid: three two-way slices through (n, p, alpha)
# ------------------------------------------------------------------
GRID = {
    "n_levels": [100, 400, 700, 1100, 1500],
    "p_levels": [100, 400, 700, 1100, 1500],
    "alpha_levels": [-8, -6, -4, -2, 0],
    "fixed_alpha": -6,
    "fixed_n": 700,
    "fixed_p": 700,
    "n_reps": 20,
    "seed": 42,              # replicate r uses seed + r for data, seed + r + 100 for the split
}

# Unit 3: parallel execution (joblib processes; each worker uses torch_threads intra-op threads)
PARALLEL = {"n_jobs": int(os.environ.get("SIM_JOBS", "-1")), "torch_threads": 1}

RUN_NAME = "sensitivity_results_prob{}_zvar{:g}".format(SPLIT["onlyA"], DGP["z_var"])

if os.environ.get("SIM_QUICK") == "1":
    GRID.update(n_levels=[60, 100], p_levels=[30, 60], alpha_levels=[-6, -2],
                fixed_alpha=-6, fixed_n=100, fixed_p=60, n_reps=2)
    MODEL["n_iter"] = 100
    EVAL["mc_samples"] = 10
    RUN_NAME += "_quick"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "results", "raw", RUN_NAME)
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "results", "figures")

BLOCKS = ["onlyX_A", "onlyX_onlyA", "onlyX_both", "A_A"]
MAIN_BLOCK = "onlyX_A"
ESTIMATORS = {"pi": "plug-in", "mc": "MC posterior mean"}
METRIC_KEYS = ["relfro_all", "relfro_avg", "relfro_cent", "rmse", "kl", "relkl", "frac_rows"]
