PYTHON ?= python3
RAW_DIR := results/raw/sensitivity_results_prob30_zvar1
SUMMARY := results/summary.csv results/trajectories.csv results/paired_mc_vs_pi.csv
FIGS    := results/figures/fig1_main_onlyX_A.pdf results/figures/fig2_training_dynamics.pdf \
           results/figures/fig3_mc_vs_plugin.pdf results/figures/fig4_heatmap_n_p.pdf

.PHONY: all simulate analyze figures test test-regression clean clean-raw simulate-quick help \
        profile complexity benchmark parallel parallel-quick stability-check perf clean-perf extensions

all: simulate analyze figures   ## full pipeline (cached scenarios are reused, not rerun)

simulate:                       ## run missing scenarios; cached pickles in $(RAW_DIR) are loaded
	$(PYTHON) simulate.py

simulate-quick:                 ## tiny grid smoke test (~10 s), writes to $(RAW_DIR)_quick
	SIM_QUICK=1 $(PYTHON) simulate.py

analyze: $(SUMMARY)             ## raw pickles -> tidy summary tables with Monte Carlo SEs
$(SUMMARY): analyze.py src/config.py $(wildcard $(RAW_DIR)/*.pkl)
	$(PYTHON) analyze.py

figures: $(FIGS)                ## summary tables -> figures (pdf + png)
$(FIGS): figures.py $(SUMMARY)
	$(PYTHON) figures.py

test:                           ## run the full test suite (includes regression tests, ~3 min)
	$(PYTHON) -m pytest -q tests

test-regression:                ## only the baseline-vs-optimised regression tests
	$(PYTHON) -m pytest -q tests/test_regression.py

# ---------------------------------------------------------------- Unit 3: performance
JOBS ?= -1

profile:                        ## cProfile + pyinstrument of one representative replicate (n=p=700)
	$(PYTHON) scripts/profile_run.py

complexity:                     ## timing vs n and vs p, baseline vs optimised, log-log slopes + plot
	$(PYTHON) scripts/complexity.py

benchmark:                      ## baseline vs optimised: per-scenario, full-study projection, parallel speedup
	$(PYTHON) scripts/benchmark.py

parallel:                       ## FULL study with the optimised code, JOBS workers (make parallel JOBS=36)
	$(PYTHON) simulate.py --jobs $(JOBS) --out results/raw/optimized_parallel

parallel-quick:                 ## same, on the tiny SIM_QUICK grid (seconds)
	SIM_QUICK=1 $(PYTHON) simulate.py --jobs $(JOBS) --out results/raw/optimized_parallel_quick

stability-check:                ## scan cached results, DGP headroom, live runs with warnings as errors
	$(PYTHON) scripts/stability_check.py

extensions:                     ## simulation-budget extension (needs benchmark output)
	$(PYTHON) scripts/budget.py

perf: profile complexity benchmark stability-check extensions   ## every Unit 3 analysis

clean:                          ## remove derived outputs (keeps the expensive raw simulation cache)
	rm -f results/*.csv results/key_numbers.json results/figures/fig*.pdf results/figures/fig*.png
	rm -rf results/raw/*_quick results/raw/optimized_parallel_quick .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-perf:                     ## remove Unit 3 timing outputs
	rm -rf results/perf results/figures/perf_*

clean-raw: clean                ## ALSO delete raw simulation results (forces a full rerun, ~hours)
	rm -rf $(RAW_DIR) results/raw/optimized_parallel

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "%-16s %s\n",$$1,$$2}'
