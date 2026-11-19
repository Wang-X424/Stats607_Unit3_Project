"""Run the simulation grid and cache raw results in results/raw/<RUN_NAME>/.

Scenarios whose pickle already exists are loaded, not rerun (use --force to rerun).
The cached results shipped with this repo were produced on the UMich cluster with the
Unit 2 (baseline) code. This script runs the optimised code (src/), in parallel by default.

    python simulate.py              # full grid (61 scenarios x 20 reps)
    SIM_QUICK=1 python simulate.py  # tiny smoke-test grid
    python simulate.py --jobs 36 --out results/raw/opt   # e.g. one Great Lakes node
"""
import argparse
import logging

from src import config
from src.simulation import build_scenarios, run_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rerun scenarios even if cached")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel worker processes (-1 = all cores, 1 = sequential; "
                         "default: config.PARALLEL / $SIM_JOBS)")
    ap.add_argument("--out", default=None, help="output folder (default: config.RAW_DIR)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("grid: %d scenarios, n_reps=%d, n_iter=%d", len(build_scenarios()),
                 config.GRID["n_reps"], config.MODEL["n_iter"])
    run_all(out_dir=args.out or config.RAW_DIR, device=args.device, force=args.force,
            n_jobs=args.jobs)
    logging.info("done")


if __name__ == "__main__":
    main()
