
SHELL := /bin/bash
PYTHON ?= python3
export PYTHONPATH := src
export TVM_NUM_THREADS := 4
export OMP_NUM_THREADS := 4
export OPENBLAS_NUM_THREADS := 1

.PHONY: smoke data models eval all

smoke:
	$(PYTHON) -m pytest -q
	$(PYTHON) scripts/smoke_harness.py
	$(PYTHON) scripts/01_collect.py --smoke

data:
	@if command -v caffeinate >/dev/null 2>&1; then \
		caffeinate -i $(PYTHON) scripts/01_collect.py; \
	else \
		$(PYTHON) scripts/01_collect.py; \
	fi
	$(PYTHON) scripts/02_split.py

models: data
	$(PYTHON) scripts/check_feature_correlations.py
	$(PYTHON) scripts/03_fit_analytic.py
	$(PYTHON) scripts/04_train.py

eval: models
	$(PYTHON) scripts/05_evaluate.py
	$(PYTHON) scripts/06_plots.py

all: smoke eval
