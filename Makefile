
SHELL := /bin/bash
PYTHON ?= python3
export PYTHONPATH := src
export TVM_NUM_THREADS := 4
export OMP_NUM_THREADS := 4
export OPENBLAS_NUM_THREADS := 1

ALLOW_DIRTY_FLAG = $(if $(ALLOW_DIRTY),--allow-dirty,)
SMOKE_DATA_DIR ?= /tmp/dvcm-study-smoke
TARGET ?= jetson_nano
FORCE_SPLIT_FLAG = $(if $(FORCE_SPLIT),--force,)

.PHONY: smoke data split models eval all

smoke:
	$(PYTHON) -m pytest -q
	$(PYTHON) scripts/smoke_harness.py
	$(PYTHON) scripts/01_collect.py --target apple_m2 --smoke --output-dir "$(SMOKE_DATA_DIR)"

data:
	@if command -v caffeinate >/dev/null 2>&1; then \
		caffeinate -i $(PYTHON) scripts/01_collect.py --target "$(TARGET)"; \
	else \
		$(PYTHON) scripts/01_collect.py --target "$(TARGET)"; \
	fi
	$(PYTHON) scripts/02_split.py --target "$(TARGET)" $(FORCE_SPLIT_FLAG)

split:
	$(PYTHON) scripts/02_split.py --target "$(TARGET)" $(FORCE_SPLIT_FLAG)

models:
	$(PYTHON) scripts/check_feature_correlations.py --target "$(TARGET)" $(ALLOW_DIRTY_FLAG)
	$(PYTHON) scripts/03_fit_analytic.py --target "$(TARGET)" $(ALLOW_DIRTY_FLAG)
	$(PYTHON) scripts/04_train.py --target "$(TARGET)" $(ALLOW_DIRTY_FLAG)

eval: models
	$(PYTHON) scripts/05_evaluate.py --target "$(TARGET)" $(ALLOW_DIRTY_FLAG)
	$(PYTHON) scripts/06_plots.py --target "$(TARGET)" $(ALLOW_DIRTY_FLAG)

all: smoke
	$(MAKE) data TARGET="$(TARGET)"
	$(MAKE) eval TARGET="$(TARGET)"
