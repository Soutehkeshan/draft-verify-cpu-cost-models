
SHELL := /bin/bash
PYTHON ?= python3
export PYTHONPATH := src
export TVM_NUM_THREADS := 4
export OMP_NUM_THREADS := 4
export OPENBLAS_NUM_THREADS := 1

ALLOW_DIRTY_FLAG = $(if $(ALLOW_DIRTY),--allow-dirty,)

.PHONY: test models eval leakage confirm report all

# Unit tests (no Nano needed).
test:
	$(PYTHON) -m pytest -q

# TRAIN-only feature check, analytic fit and XGBoost training on data/jetson_nano/.
models:
	$(PYTHON) scripts/check_feature_correlations.py $(ALLOW_DIRTY_FLAG)
	$(PYTHON) scripts/02_fit_analytic.py $(ALLOW_DIRTY_FLAG)
	$(PYTHON) scripts/03_train.py $(ALLOW_DIRTY_FLAG)

# Every scorer on the held-out TEST schedules, into results/jetson_nano/.
eval: models
	$(PYTHON) scripts/04_evaluate.py $(ALLOW_DIRTY_FLAG)

# Shuffled-label leakage check (120 permutations; trains 240 models, slow).
leakage:
	$(PYTHON) scripts/check_shuffled_label_null.py $(ALLOW_DIRTY_FLAG)

# Frozen models on the confirmation TEST set in data/jetson_nano_confirm/.
confirm:
	$(PYTHON) scripts/evaluate_confirmation.py $(ALLOW_DIRTY_FLAG)

# Report tables and figures from the result files, then check every number in Report/*.tex.
report:
	$(PYTHON) scripts/make_report_tables.py
	$(PYTHON) scripts/make_report_figures.py
	$(PYTHON) scripts/check_report_numbers.py

all: eval leakage report
