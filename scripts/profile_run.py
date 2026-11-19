"""Profile one representative replicate (n = p = 700, alpha = -6) with cProfile and pyinstrument.

    python scripts/profile_run.py [--n-iter 300] [--version both|baseline|optimized]

Writes results/perf/profile_<version>.txt (cProfile, sorted by self time) and
results/perf/profile_<version>.html (pyinstrument call tree, if installed).
A short warm-up replicate runs first so import/BLAS-initialisation costs are excluded.
"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402
from src.baseline.simulation import run_one_rep as run_base  # noqa: E402
from src.simulation import run_one_rep as run_opt  # noqa: E402

OUT = os.path.join(config.RESULTS_DIR, "perf")
CFG = {"label": "N700_X700_Aneg6", "N_NODES": 700, "X_DIM": 700, "Alpha": -6}


def profile(version, n_iter):
    run = run_base if version == "baseline" else run_opt
    mcfg = dict(config.MODEL, n_iter=n_iter)
    run(CFG, 1, model_cfg=dict(mcfg, n_iter=20))  # warm-up

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    run(CFG, 0, model_cfg=mcfg)
    pr.disable()
    wall = time.perf_counter() - t0
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf).strip_dirs()
    buf.write(f"# {version}: n=p=700, alpha=-6, {n_iter} iterations, wall {wall:.2f}s\n")
    st.sort_stats(pstats.SortKey.TIME).print_stats(20)
    st.sort_stats(pstats.SortKey.CUMULATIVE).print_stats("methods.py|metrics.py|simulation.py|dgps.py", 15)
    with open(os.path.join(OUT, f"profile_{version}.txt"), "w") as f:
        f.write(buf.getvalue())
    print(f"{version}: {wall:.2f}s -> results/perf/profile_{version}.txt")

    try:
        from pyinstrument import Profiler
    except ImportError:
        print("pyinstrument not installed; skipping call-tree profile")
        return
    prof = Profiler(interval=0.001)
    prof.start()
    run(CFG, 0, model_cfg=mcfg)
    prof.stop()
    with open(os.path.join(OUT, f"profile_{version}.html"), "w") as f:
        f.write(prof.output_html())
    with open(os.path.join(OUT, f"pyinstrument_{version}.txt"), "w") as f:
        f.write(prof.output_text(unicode=False, color=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--version", default="both", choices=["both", "baseline", "optimized"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for v in (["baseline", "optimized"] if a.version == "both" else [a.version]):
        profile(v, a.n_iter)
