import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Tiny configuration shared by the tests that train the model.
TINY_GRID = {"seed": 42, "n_reps": 1}
TINY_MODEL = {"z_dim_fit": 8, "h_dim_fit": 32, "lr": 1e-3, "lr_min": 1e-5, "l2": 1e-2,
              "self_weight": 1.0, "tau2": 1.0, "n_iter": 60}
TINY_EVAL = {"mc_samples": 5, "mc_every_log": False, "min_exp_deg": 1.0, "log_every": 50}
TINY_CFG = {"label": "tiny", "N_NODES": 60, "X_DIM": 30, "Alpha": -2}
