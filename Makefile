
SHELL := /bin/bash
PYTHON ?= python3
export PYTHONPATH := src
export TVM_NUM_THREADS := 4
export OMP_NUM_THREADS := 4
export OPENBLAS_NUM_THREADS := 1

ALLOW_DIRTY_FLAG = $(if $(ALLOW_DIRTY),--allow-dirty,)
SMOKE_DATA_DIR ?= /tmp/dvcm-study-smoke

.PHONY: smoke data models eval all

smoke:
	$(PYTHON) -m pytest -q
	$(PYTHON) scripts/smoke_harness.py
	$(PYTHON) scripts/01_collect.py --smoke --output-dir "$(SMOKE_DATA_DIR)"

data:
	@if command -v caffeinate >/dev/null 2>&1; then \
		caffeinate -i $(PYTHON) scripts/01_collect.py; \
	else \
		$(PYTHON) scripts/01_collect.py; \
	fi
	$(PYTHON) scripts/02_split.py

models:
	@test -f data/pool.parquet -a -f data/splits.json || \
		(echo "Run 'make data' first to create the study pool and splits." >&2; exit 1)
	$(PYTHON) scripts/check_feature_correlations.py $(ALLOW_DIRTY_FLAG)
	$(PYTHON) scripts/03_fit_analytic.py $(ALLOW_DIRTY_FLAG)
	$(PYTHON) scripts/04_train.py $(ALLOW_DIRTY_FLAG)

eval: models
	$(PYTHON) scripts/05_evaluate.py $(ALLOW_DIRTY_FLAG)
	$(PYTHON) scripts/06_plots.py $(ALLOW_DIRTY_FLAG)

all: smoke
	$(MAKE) data
	$(MAKE) eval
