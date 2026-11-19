"""Reproducibility, fidelity to the original script, and validity of the cached raw results."""
import ast
import glob
import os
import pickle
import sys

import numpy as np
import pytest
import torch

from src import config
from src.baseline.simulation import run_one_rep as run_one_rep_baseline
from src.simulation import build_scenarios, run_one_rep

from .conftest import ROOT, TINY_CFG, TINY_EVAL, TINY_GRID, TINY_MODEL


def _run_tiny():
    return run_one_rep(TINY_CFG, 0, grid=TINY_GRID, model_cfg=TINY_MODEL, eval_cfg=TINY_EVAL)


def test_design_has_61_unique_scenarios():
    scen = build_scenarios()
    if os.environ.get("SIM_QUICK") == "1":
        pytest.skip("quick grid")
    keys = {(c["N_NODES"], c["X_DIM"], c["Alpha"]) for c in scen}
    assert len(scen) == len(keys) == 61


def test_same_seed_gives_identical_results():
    a, b = _run_tiny(), _run_tiny()
    assert len(a) == len(b) == 2  # iterations 0 and 49
    for ra, rb in zip(a, b):
        for k in ra:
            assert (np.isnan(ra[k]) and np.isnan(rb[k])) or ra[k] == rb[k], k


def _load_legacy():
    """Execute only the definitions/constants of legacy/simulation_sensitivity_fix1_prob.py."""
    path = os.path.join(ROOT, "legacy", "simulation_sensitivity_fix1_prob.py")
    tree = ast.parse(open(path).read())
    skip = {"percents", "PROP_ONLYA", "PROP_BOTH", "PROP_ONLYX", "SENS_DIR", "SENSITIVITY_SCENARIOS", "NITER_LOG", "seen", "device"}
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            keep.append(node)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if not names & skip:
                keep.append(node)
    ns = {"__name__": "legacy"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    ns.update(device=torch.device("cpu"), PROP_ONLYA=30, PROP_BOTH=40, PROP_ONLYX=30,
              MC_SAMPLES=TINY_EVAL["mc_samples"], H_DIM_FIT=TINY_MODEL["h_dim_fit"])
    return ns


def test_refactor_reproduces_original_script_exactly():
    """With the torch RNG seeded identically, the Unit 2 refactor (frozen in src/baseline)
    returns the same numbers as the original server script, so the cached results correspond
    to that code. Baseline-vs-optimised agreement is tested in tests/test_regression.py."""
    legacy = _load_legacy()
    torch.manual_seed(TINY_GRID["seed"])
    old = legacy["run_one_scenario"](TINY_CFG, n_reps=1, n_iter=TINY_MODEL["n_iter"],
                                     seed_base=TINY_GRID["seed"])[0]
    new = run_one_rep_baseline(TINY_CFG, 0, grid=TINY_GRID, model_cfg=TINY_MODEL, eval_cfg=TINY_EVAL)
    assert [r["iter"] for r in old] == [r["iter"] for r in new]
    for ro, rn in zip(old, new):
        for k in ro:
            if np.isnan(ro[k]):
                assert np.isnan(rn[k]), k
            else:
                assert rn[k] == pytest.approx(ro[k], rel=1e-6, abs=1e-9), k


RAW = sorted(glob.glob(os.path.join(config.RAW_DIR, "sens_*.pkl")))


@pytest.mark.skipif(not RAW, reason="no cached raw results")
def test_cached_raw_results_are_complete_and_valid():
    labels = {os.path.basename(p)[5:-4] for p in RAW}
    assert labels == {c["label"] for c in build_scenarios()}
    for path in RAW:
        with open(path, "rb") as f:
            res = pickle.load(f)
        assert len(res) == config.GRID["n_reps"]
        for rep in res:
            last = rep[-1]
            assert last["iter"] == config.MODEL["n_iter"] - 1
            for k, v in last.items():
                if k.startswith(("pi_", "mc_")) and not k.startswith(("pi_relfro_avg", "mc_relfro_avg")):
                    assert np.isfinite(v) and v >= 0, (path, k)
                if "frac_rows" in k:
                    assert 0 <= v <= 1
            # the MC estimator is only evaluated at the final iteration
            assert np.isnan(rep[0]["mc_relkl_onlyX_A"]) and np.isfinite(last["mc_relkl_onlyX_A"])
