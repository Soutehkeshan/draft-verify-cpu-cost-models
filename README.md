# Reproducible CPU Cost-Model Study

This repository compares random, analytical, and learned schedule rankers for
TVM MetaSchedule CPU matrix multiplication (`N=512`) on Apple M2 arm64. It also
measures an analytic-draft / XGBoost-verify pipeline. Collection is uniform over
valid encoded tilings and unroll candidates; this differs from TVM's default
replay sampling.

## Setup

Use the Python version supported by the pinned TVM wheel, install LLVM and its
development headers if required by that wheel, then install the exact Python
dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default target is `llvm -mcpu=apple-m2 -num-cores=4`. `make` sets
`TVM_NUM_THREADS=4`, `OMP_NUM_THREADS=4`, and `OPENBLAS_NUM_THREADS=1`.

## Commands

```sh
make smoke  # pytest, 10 harness measurements, and a 30-schedule smoke dataset
make data   # collect 5,000 schedules (or 3,000 if the pilot projects >8 hours)
make models # split, fit analytic constants on TRAIN, tune and train XGBoost
make eval   # test-only metrics and PDF plots
make all    # smoke, full collection, models, and evaluation
```

On macOS, `make data` wraps collection with `caffeinate -i` when available.
For a reproducible collection run, keep the machine plugged in, turn Low Power
Mode off, and use the documented four TVM threads. macOS does not support
pinning the process to specific CPU cores, which remains a source of timing
variance. The smoke target is intended to finish in under five minutes. Full
collection is hardware-dependent: the pilot reports a projection, caps the
default pool at 3,000 when that estimate exceeds eight hours, and logs
throughput and ETA. Model fitting and plotting run after collection and their
runtime depends on the resulting pool and XGBoost performance.

## Data and outputs

Generated datasets are stored in [`data/`](./data/):

- `pool.parquet`: one row per schedule; `id`, `sketch_id`, integer `vector`,
  repeat timings in milliseconds, median and standard deviation, status,
  build duration, batch, and UTC timestamp.
- `layout_<sketch_id>.json`: JSON-only shape, tiling, and unroll-candidate
  description for that design-space sketch.
- `run_meta.json`: toolchain, machine, target, run configuration, timing cap,
  timestamps, and status totals.
- `splits.json`: seeded, per-sketch 70/15/15 schedule-ID splits and nested
  TRAIN subsets.
- `canary.json`: fixed schedules remeasured at batch boundaries.

The `make smoke` collection is isolated under `data/smoke/`, so its two-repeat
measurements never mix with the five-repeat full study.

Rows with `ok` and `slow_skip` status are modelable; a slow-skip row carries its
single-run measurement and is noisier. Failure statuses remain in the parquet
and their rates are written to `results/failure_rates.csv`, never silently
discarded from the run record. If slow-skip rows exceed 5%, treat model rankings
as lower-confidence and call this out when interpreting results. Run metadata
reports the observed fraction.

Evaluation CSVs and PDFs are written under [`results/`](./results/). The
generated dataset is the study artifact; no precomputed measurements are
checked into this repository.

## Method and references

The analytical latency model follows the compute-plus-memory structure of
Pruner (Qiao et al., ASPLOS 2025). The schedule search context uses TVM
MetaSchedule's LLVM design-space rules and postprocessors. Results include a
test-set-only comparison, training-size curves, bootstrap 95% confidence
intervals, ablations, measurement noise floor, and draft-then-verify budget
curves.

- Qiao et al., “Pruner,” ASPLOS 2025.
- Zheng et al., “Ansor: Generating High-Performance Tensor Programs for Deep
  Learning,” OSDI 2020, [USENIX paper page](https://www.usenix.org/conference/osdi20/presentation/zheng).

The derived feature interface is explicit and deterministic: per-level tile
footprints sum the float32 A, B, and C working sets; FLOPs are `2*N^3`;
estimated bytes moved account for innermost I/J reuse and the C write;
vector utilization is the innermost I factor capped by lane width; and
parallel utilization is the outer I/J tile count capped by four cores. These
are study heuristics, not hardware-counter measurements.

## Reproducibility

All NumPy, Python, TVM, model, data-split, and bootstrap randomness is seeded
with `SEED=0` or a deterministic seed derived from it. TVM's schedule
constructor rejects a zero seed, so schedule replay maps that value to its
smallest valid seed (1); MetaSchedule's design-space RNG remains seeded with
zero. Measurements are
machine-specific; for each collection, retain `data/run_meta.json`,
`data/layout_*.json`, `data/splits.json`, and the exact `pool.parquet`
together. `make all` regenerates the data if needed and then completes the
analysis without manual code edits.