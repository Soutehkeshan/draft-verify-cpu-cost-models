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
python scripts/01_collect.py --pilot-only  # real-settings pilot and 5,000-sample projection
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
- `submission_bundle.zip`: convenience archive containing the measured pool,
  split and run metadata, and all per-sketch layout descriptions.

The `make smoke` collection is isolated under `data/smoke/`, so its two-repeat
measurements never mix with the five-repeat full study.

Rows with `ok` and `slow_skip` status are modelable; a slow-skip row carries its
single-run measurement and is noisier. Failure statuses remain in the parquet
and their rates are written to `results/failure_rates.csv`, never silently
discarded from the run record. If slow-skip rows exceed 5%, treat model rankings
as lower-confidence and call this out when interpreting results. Run metadata
reports the observed fraction.

Evaluation CSVs and PDFs are written under [`results/`](./results/). The
generated measurements are intentionally excluded from Git. Before submission,
publish `pool.parquet`, `splits.json`, and `run_meta.json` together as a
downloadable release asset and replace this sentence with its public URL.
`data/` contains the local run used for the reported results.

`results/evaluation.csv` reports all five scorers pooled and separately for
each sketch. Its Top-k latency ratio is the mean measured latency among the
scorer's selected top-k schedules divided by the fastest measured latency in
that test scope; lower is better and 1.0 is perfect. Thus a Top-1 ratio of 1.45
means the chosen schedule is 45% slower than the test-set oracle. The table
also includes Spearman within the fastest 10% of schedules, and recall of the
true ten fastest schedules within the scorer's top 30. Bootstrap intervals
resample held-out schedules. The full pooled and per-sketch table is also
written to CSV so the baseline scorers can be compared side by side.

`results/noise_floor.csv` reports repeat-half Spearman overall and within the
fastest decile, by sketch and pooled. The overall value estimates a high
measurement-repeatability ceiling; the top-decile value is lower because
rankings among the fastest candidates are more timing-sensitive.
`results/feature_comparison_seeds.csv` contains the five paired XGBoost
base-versus-hybrid runs and per-run bootstrap intervals;
`feature_comparison_summary.csv` reports across-seed 95% t intervals.
`learning_curve.csv` contains all five runs per nested training subset and
`learning_curve_summary.csv` summarizes them with across-seed intervals.
`shuffled_label_check.csv` records the fixed-seed control that retrains after
permuting TRAIN labels once. Analytic constants are fitted on TRAIN only, and
the XGBoost grid is selected on VALIDATION only.

## Current held-out results

The test split contains 752 schedules; 76 are in its fastest decile. Ratios
below are chosen latency divided by the best measured latency (lower is better).
Top-1 intervals are bootstrap 95% CIs over test schedules.

| Scorer | Spearman | Fastest-decile Spearman | True top-10 recall in top 30 | Top-1 ratio (95% CI) |
|---|---:|---:|---:|---:|
| Random | 0.032 | 0.166 | 0.20 | 2.148 (1.986-14.096) |
| Analytic (hand-set) | 0.457 | -0.044 | 0.00 | 1.750 (1.306-8.718) |
| Analytic (fitted) | 0.484 | -0.050 | 0.00 | 1.750 (1.306-8.718) |
| XGBoost (base) | 0.894 | 0.071 | 0.40 | 1.616 (1.073-1.616) |
| XGBoost (base + analytic) | 0.903 | 0.235 | 0.70 | 1.449 (1.221-1.616) |

Per-sketch results for XGBoost (base + analytic) are:

| Sketch | Test schedules | Spearman | Fastest-decile Spearman | Top-10 recall in top 30 | Top-1 ratio |
|---:|---:|---:|---:|---:|---:|
| 0 | 240 | 0.884 | -0.097 | 0.70 | 1.335 |
| 1 | 246 | 0.907 | 0.410 | 0.80 | 1.231 |
| 2 | 266 | 0.911 | 0.769 | 1.00 | 1.284 |

Repeat-half Spearman is 0.986 pooled, while it is 0.730 within the fastest
decile (75 repeatable schedules); the top-decile timing ranks are less stable
than the full-pool ranks. Across five stochastic model seeds, adding analytic
features improved mean Spearman by 0.013 (95% t CI 0.008-0.018), a modest
difference below 0.02. The paired Top-1 ratio difference was -0.126 (95% t CI
-0.359 to 0.107), which does not establish a Top-1 improvement. A model trained
once on shuffled TRAIN labels had test Spearman 0.094 and fastest-decile
Spearman -0.003, consistent with no meaningful ranking signal.

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

## Derived features and analytic model

For each sampled position, the i, j, and k exponents define split-loop tile
sizes `I_l=2**i_l`, `J_l=2**j_l`, and `K_l=2**k_l` elements (the exponents
are dimensionless). The current LLVM sketch uses the `SSRSRS` order below;
the indices are loop counters with the corresponding split-factor trip
counts:

```text
for i0 -> j0 -> i1 -> j1 -> k0 -> i2 -> j2:
  for k1:             # reduction loop; eligible for auto-unroll under U
    for i3 -> j3:     # innermost output tile; j3 is vectorized
      C[i,j] += A[i,k] * B[k,j]
```

In terms of loop extents, the order is
`i0 -> j0 -> i1 -> j1 -> k0 -> i2 -> j2 -> k1 -> i3 -> j3`.
`i3 x j3` is the innermost output tile, with `k1` as the innermost reduction
level; `j3` is the contiguous vectorized loop. For example, a tiny-tile
position uses i/j factors `(512,1,1,1)`, k factors `(512,1)`, and `U=0`
(no explicit unrolling). A large-tile position uses i/j `(1,1,8,64)`, k
`(8,64)`, and `U=64`; its 64-trip `k1` reduction loop is eligible for
auto-unrolling. The unroll choice `U` is the root
`meta_schedule.unroll_explicit` annotation (candidates `0,16,64,512`): it
sets the limit for the postprocessor, which selects eligible loops rather
than encoding one fixed loop to unroll.

The deterministic derived descriptors use these formulas, where
`M=product(I_l)`, `N=product(J_l)`, `K=product(K_l)`, and an FP32 element is
4 bytes:

| Descriptor | Formula | Units | Hardware input |
|---|---|---|---|
| Level-l working-set footprint `F_l`, for `l=0..3` | `4*(I_{l:3}*K_{max(0,l-1):1} + K_{max(0,l-1):1}*J_{l:3} + I_{l:3}*J_{l:3})` | bytes | 4 bytes/element |
| Estimated DRAM/L2-boundary traffic | `I_0*J_0*F_1` | bytes | 4 bytes/element |
| FLOPs | `2*M*N*K` | FLOPs | none |
| Vector-lane utilization | `min(J_3,W)/W` | dimensionless, `[0,1]` | lane width `W` |
| Parallel utilization | `P/(ceil(P/C)*C)`, `P=product of the i/j tile factors in the sketch's parallel loop prefix` | dimensionless, `[0,1]` | core count `C` |
| Register utilization | `min(1,R*W/(I_3*J_3))` | dimensionless, `[0,1]` | 32 architectural SIMD registers `R`, lane width `W` |

Here `I_{a:b}` and `K_{a:b}` denote products of the factors from level `a`
through `b`, inclusive; likewise for `J`. For `F_0` and `F_1`, the suffix
contains both reduction loops. At `F_2`, `k0` is outside the region and only
`k1` remains; at `F_3`, both reduction loops are outside, so the k extent is
one. All three sketches have the same `SSRSRS` order, but their cache-write
placements give different parallel prefixes: sketches 0 and 2 use i0,i1,j0,j1;
sketch 1 uses i0,j0. The layout records this structure from the trace.

`I_0*J_0*F_1` estimates bytes crossing the DRAM/L2 boundary: each region
selected by i0,j0 transfers its F1 working set once, with reuse inside that
region but no reuse assumed between regions. This is a boundary-traffic
estimate, not register-level traffic and not a hardware-counter measurement.
The tiny- and large-tile examples both have `2*512**3 = 268,435,456` FLOPs;
their estimated traffic is respectively `1,074,790,400` and `3,145,728`
bytes. `F_0` and FLOPs are invariant for this fixed-size workload, so they
remain in `derived()` for validation/correlation but are excluded from the
learned feature vector. The aggregate sum of nested footprints and a separate
unroll-value feature are omitted; the encoded unroll index already represents
that choice.

The analytic estimate is Pruner-style additive compute plus memory time, in
seconds (not roofline's max):

```text
compute_time = FLOPs / (peak_flops * P_vector * P_parallel * P_register)
memory_time  = bytes_moved / (DRAM_bandwidth * P_L1 * P_L2)
```

Every penalty is in `[0,1]` and equals 1 when no modeled limit is active.
`P_vector=min(J_3,W)/W` is below 1 when the innermost j tile is narrower than
the SIMD lane width. `P_parallel=P/(ceil(P/C)*C)` accounts for the partially
filled final batch of parallel tasks. `P_register=min(1,R*W/(I_3*J_3))` is
below 1 when the innermost output tile needs more accumulator vectors than
the register budget. The model uses the 32 architectural AArch64 SIMD
registers as an upper-bound budget; real kernels have fewer available after
other live values. The working-set penalties map to their corresponding
levels: `P_L1=min(1,L1_bytes/F_2)` and `P_L2=min(1,L2_bytes/F_1)`. They
represent effective-memory-throughput penalties when those working sets
exceed the respective cache capacities. Vector, parallel, and register
penalties affect compute time; cache penalties affect memory time.

Only `peak_flops` and `bandwidth_bytes_s` are fitted, using TRAIN measurements.
Their starting points are 400 GFLOP/s and 100 GB/s; `results/constants.json`
records the fitted-to-initial multiplier so the fit movement is explicit.
Hand-set assumptions are FP32 (4 bytes/element), L1 data cache (128 KiB),
performance-cluster L2 (16 MiB), SIMD lane width (4 float32 lanes), four
cores, and the 32-register architectural budget. The cache sizes were read
from this host using `sysctl hw.perflevel0.l1dcachesize` and
`sysctl hw.perflevel0.l2cachesize`; there is no separate L3 penalty. The
100 GB/s DRAM bandwidth is a nominal M2 starting value, not a local measured
bandwidth. `make models` first waits for collection and creates the split,
then writes `results/feature_correlations.csv` with each descriptor's
Spearman correlation against latency on TRAIN only. Descriptors with fewer
than two distinct values or `abs(rho) < 0.02` are marked `flat`; expected-sign
conflicts are marked `rethink`. In the completed 3,499-row TRAIN split, FLOPs
and F0 are invariant, F1 is flat (`rho=-0.008`), and register utilization has
a weak opposite-direction correlation (`rho=+0.032`, `p=0.055`). Those
descriptors are excluded from learned features; register pressure remains an
explicit analytic-model hypothesis for ablation. F2, F3, bytes moved, vector
utilization, and parallel utilization remain model features. The report
includes `included_in_model` so each diagnostic's treatment is explicit.

For that same fit, peak FLOP/s moved from 400 to 180.4 GFLOP/s (0.451x) and
DRAM bandwidth moved from 100 to 94.7 GB/s (0.947x). The fit multipliers and
exact values are written to `results/constants.json`.

## Reproducibility

All NumPy, Python, TVM, model, data-split, and bootstrap randomness is seeded
with `SEED=0` or a deterministic seed derived from it. Five-seed XGBoost runs
use seeded 80% row and column sampling so the reported seed intervals reflect
distinct reproducible fits. TVM's schedule
constructor rejects a zero seed, so schedule replay maps that value to its
smallest valid seed (1); MetaSchedule's design-space RNG remains seeded with
zero. Measurements are
machine-specific; for each collection, retain `data/run_meta.json`,
`data/layout_*.json`, `data/splits.json`, and the exact `pool.parquet`
together. `make all` regenerates the data if needed and then completes the
analysis without manual code edits.