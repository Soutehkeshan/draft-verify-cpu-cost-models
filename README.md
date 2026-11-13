# CPU Cost-Model Study: Jetson Nano Evaluation Scaffold

## Research question

**How effectively can analytical and machine-learning cost models predict CPU
schedule performance on an NVIDIA Jetson Nano, and does adding analytical
features improve XGBoost schedule ranking?**

The study is intentionally limited to one workload: FP32 matrix multiplication
with `M=N=K=512`. It compares an analytical model, XGBoost on schedule-derived
features, and hybrid XGBoost on schedule-derived and analytical features.
Random ranking is the ranking baseline. The project is CPU-only; CUDA and GPU
scheduling are out of scope.

## Current status

**HOST-SIDE REFACTORING COMPLETE — WAITING FOR JETSON NANO INTEGRATION**

No Jetson Nano compilation, collection, or measurement has been performed or
validated. The Nano profile records only the supplied hardware and software
facts. Its TVM target, schedule rules, postprocessors, SIMD/cache/register
values, peak compute, and memory bandwidth remain unset; the collection command
refuses to run until the compiler profile is explicitly completed.

The prior Apple M2 data release and its results are retained as historical,
M2-only material. They are not Jetson measurements and are not presented as
evidence for the Nano research question.

## Scope and hardware profiles

Profiles are stored in [`configs/hardware/`](./configs/hardware/), separate
from the model implementations.

| Profile | Known configuration | Status |
|---|---|---|
| `jetson_nano` | NVIDIA Jetson Nano 4GB Developer Kit; ARM Cortex-A57; four CPU cores; reported Ubuntu 18.04.5, aarch64, L4T R32.5.0 / JetPack 4.5, GCC 7.5.0, Python 3.6.9. CUDA 10.2 is recorded as installed but is not used. | Primary target; host-side profile only. Hardware-specific compiler and model assumptions are deliberately unset. |
| `apple_m2` | MacBook Air Apple M2, arm64, four cores; existing TVM LLVM target and M2 study assumptions. | Compatibility profile for the previously collected M2 dataset and local smoke tests. |

The M2 profile's 400 GFLOP/s and 100 GB/s values are legacy analytical-fit
starting assumptions, not measured hardware peaks. Its cache and vector
assumptions are retained only to reproduce the earlier M2 work. Nano peak
compute and bandwidth must be calibrated from TRAIN measurements (or supplied
with a documented source); unknown Nano values are never filled with M2
defaults.

The reported GCC version does not identify the LLVM backend or TVM compiler
target. In particular, this repository does not establish a TVM/LLVM build
compatible with the Nano's Python 3.6.9 environment. The current pinned host
dependencies and Python syntax require Python 3.10 or newer; Nano
compatibility must be resolved before deployment. Do not upgrade or change the
Nano's system software as part of this host-side work.

## Cost models

1. **Analytical:** estimates latency as compute time plus memory time. The
   compute term uses FLOPs and peak compute; the memory term uses estimated
   bytes crossing the modeled memory boundary and bandwidth. Vector,
   parallel-batch, register, and cache penalties are explicit. The cache and
   compute/bandwidth parameters are selected per hardware profile. Peak compute
   and bandwidth fitting uses TRAIN measurements only.
2. **XGBoost:** predicts log latency from the encoded schedule exponents,
   unroll-candidate index, and sketch identifier.
3. **Hybrid XGBoost:** uses the XGBoost schedule features plus the derived
   schedule descriptors and analytical score.

XGBoost hyperparameters are selected on VALIDATION only. Final metrics use the
same held-out TEST schedules for every scorer. The existing evaluation script
also emits a random-ranking baseline and diagnostics; it does not fit
parameters or select features from TEST measurements.

## Workload, schedules, and features

The workload is configured in
[`configs/workloads/matmul_fp32_512.json`](./configs/workloads/matmul_fp32_512.json):
512-by-512 FP32 matrix multiplication, seed 0, with five timing repeats in the
full-study protocol. The existing TVM MetaSchedule encoding represents valid
tile-exponent combinations and an unroll candidate index. Collection samples
uniformly over valid encoded schedules per sketch; this differs from TVM's
default replay sampling. Schedule IDs are deduplicated by `(sketch_id, vector)`.

Features retained from the M2 implementation include tile exponents,
unroll-candidate index, and sketch ID. Derived schedule descriptors include
tile working-set footprints, FLOPs, estimated DRAM/L2-boundary traffic,
vector-lane utilization, parallel utilization, and register utilization.
Register utilization is used by the analytical penalty; it is not in the
current learned feature vector. Traffic is an analytical estimate, **not** a
hardware-counter measurement. Hardware-dependent features require values from
the selected profile; the feature code does not silently assume M2 values for
another target.

## Architecture

- `configs/hardware/`: target identity, known hardware values, compiler and
  schedule configuration, and target-specific data/result paths.
- `configs/workloads/`: workload shape, dtype, seed, and measurement defaults.
- `src/dvcm/config.py`: profile loading and validation.
- `src/dvcm/encoding.py`: encode, repair, and replay schedule decisions.
- `src/dvcm/features.py`: target-independent schedule descriptors and
  explicitly target-bound utilization features.
- `src/dvcm/analytic.py`: analytical latency model and TRAIN-only constant fit.
- `src/dvcm/learned.py`: XGBoost training and prediction.
- `src/dvcm/harness.py`: workload construction, schedule replay, compilation,
  correctness checking, and timing for an explicitly selected TVM target.
- `src/dvcm/study.py`: dataset loading, split validation, and deterministic
  split generation.
- `scripts/`: collection, splitting, fitting, training, evaluation, and plots.
- `tests/`: synthetic unit tests plus host TVM schedule/build smoke tests.

The harness takes a target and postprocessor configuration rather than
selecting Apple M2 internally. However, the available compiler profile is
currently tested only for the existing M2 development setup. This interface is
not a claim that the Nano can compile or run the schedules.

## Data and experiment methodology

The existing pool is a Parquet table with one row per encoded schedule:

| Field | Meaning |
|---|---|
| `id`, `sketch_id`, `vector` | Schedule identity and encoded decisions. |
| `repeats_ms` | Individual execution measurements in milliseconds. |
| `median_ms`, `std_ms` | Aggregated timing and repeat variability. |
| `status` | `ok`, `slow_skip`, `build_fail`, or `wrong_result`. |
| `build_time_s`, `batch_id`, `timestamp` | Build duration, collection batch, and UTC sample time. |

`layout_<sketch_id>.json` describes schedule-space extents and encoded
decisions. `run_meta.json` records the selected target/compiler configuration, available
runtime versions, workload, measurement settings, thread configuration,
timestamps, and status totals. An unknown compiler version remains unset.
`splits.json` records schedule-ID partitions and nested TRAIN subsets.
Trained models, constants, evaluation tables, and plots are separate generated
outputs. CSV outputs have metadata companions; JSON outputs record provenance.

Target paths prevent new Nano artifacts from mixing with the historical M2
pool: the M2 compatibility profile reads the existing `data/` pool and writes
new results to `results/apple_m2/`; the Nano profile reserves `data/jetson_nano/`
and `results/jetson_nano/`. Existing M2 result artifacts in the root
`results/` are left untouched. Generated data and results remain ignored by
Git. The released M2 dataset is available at
[`study-data-v1`](https://github.com/Soutehkeshan/draft-verify-cpu-cost-models/releases/tag/study-data-v1)
and as the
[dataset bundle](https://github.com/Soutehkeshan/draft-verify-cpu-cost-models/releases/download/study-data-v1/submission_bundle.zip).

The split is seeded and stratified by sketch ID: 70% TRAIN, 15% VALIDATION,
and 15% TEST. Splits are disjoint and exhaustive over modelable measurements.
The analytic constants are fit using TRAIN labels only. The XGBoost grid is
selected using VALIDATION labels only, after which the final models are trained
on the designated TRAIN subsets. TEST labels are reserved for reporting.
Repeated timings are kept in the pool; the evaluator uses `median_ms` as its
schedule-level target.

## Evaluation metrics

- **Spearman:** rank correlation between predicted and measured latency.
- **Top-10 recall in 30:** fraction of the ten measured-fastest TEST schedules
  present among the model's 30 lowest predicted latencies. The candidate set is
  explicitly 30, not 10.
- **Top-1 latency ratio:** measured latency of the model-selected schedule
  divided by the fastest measured TEST latency. A ratio of 1.0 is optimal;
  lower is better.
- **Measurement count:** the number of held-out schedules is recorded as
  `n_test`; where a candidate budget applies, that budget is reported
  separately.

The evaluator also retains the prior study's top-k ratios, fastest-decile
Spearman, bootstrap intervals, paired base-versus-hybrid comparisons,
shuffled-label checks, ablations, and draft/verify diagnostics. These do not
change the train/validation/test roles.

## Installation and host commands

Use the pinned dependencies on a compatible development-machine Python. The
current dependency set is not certified for Python 3.6.9; do not install it on
the Nano as part of this task.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make smoke
```

The smoke target runs tests, then compiles and measures a small number of
schedules with the explicit M2 compatibility profile. It is a host integration
check, not an experiment or a Nano validation.
The Makefile exports `TVM_NUM_THREADS=4`, `OMP_NUM_THREADS=4`, and
`OPENBLAS_NUM_THREADS=1`. On macOS, collection uses `caffeinate -i` when
available; keep the machine plugged in and Low Power Mode off. macOS cannot
pin the process to specific CPU cores.

For the existing M2 dataset:

```sh
make split TARGET=apple_m2
make models TARGET=apple_m2
make eval TARGET=apple_m2
make all TARGET=apple_m2
```

An existing valid split is preserved by default. To intentionally regenerate
it after changing the pool, use `make split TARGET=apple_m2 FORCE_SPLIT=1`.
Training/evaluation outputs go under `results/apple_m2/`; they do not overwrite
the old root-level result snapshot. Full M2 collection uses
`make data TARGET=apple_m2` and is a separate, machine-dependent operation.
Result-producing scripts require a clean committed Git worktree by default.
Commit first, or explicitly use `ALLOW_DIRTY=1`; dirty-run provenance is
recorded when that override is used.

The default `TARGET` is `jetson_nano`, because it is the primary research
target. Nano collection currently stops with an explicit unconfigured-target
error before creating target data or result directories. Do not interpret that
profile or failure as Nano benchmark support.

## What is tested and what remains

**Tested locally:** profile loading and validation; target-bound feature
requirements; analytic estimation and fitting on synthetic test inputs;
XGBoost training on synthetic test inputs; deterministic split validation;
ranking metrics; TVM M2 schedule generation, schedule postprocessing, compile,
correctness, and smoke timing. No synthetic measurements are used as
experimental results.

**Existing M2 evidence:** the released M2 dataset and historical M2 result
artifacts. These are not re-labeled or reinterpreted as Nano measurements.

**Awaiting Nano integration:** compatible TVM and compiler availability,
validated CPU target and schedule rules, verified postprocessors, cache/SIMD/
register assumptions, training-calibrated compute/bandwidth constants, and
actual Nano collection, training, and test evaluation. No Nano performance
results are reported.

## Handoff: Nano-specific work

Stop here until the hardware owner reviews and approves Nano integration. The
remaining work is:

1. On the Nano, verify its OS, architecture, GCC/Python versions, and the
   already-installed TVM/LLVM compiler versions without installing or upgrading
   anything.
2. Confirm that the existing TVM build can target and execute CPU code on the
   Cortex-A57, and identify the correct target object, schedule rules, and
   postprocessors. Only then populate `compiler_target`, `schedule_rules`, and
   `postprocessors` in `configs/hardware/jetson_nano.json`.
3. Verify SIMD width, cache sizes, and any register assumptions from
   authoritative system or architecture information. Do not copy M2 values.
4. Resolve the Python 3.6.9/dependency compatibility question before attempting
   to run this repository on the device. No system-package or JetPack changes
   are authorized by this refactor.
5. After configuration and compatibility are approved, run a single known-good
   CPU schedule for correctness, then a small measured pilot. Only after those
   checks should the full target-specific collection and evaluation begin.

## Limitations and references

The current harness is still based on the repository's existing TVM
MetaSchedule implementation and version; it has not been ported or validated
against the Nano's older software stack. Analytical features are estimates and
omit hardware counters. The current workload is only one FP32 matrix size.
Timing quality depends on device power, thermal, and scheduling conditions.

- Qiao et al., “Pruner,” ASPLOS 2025.
- Zheng et al., “Ansor: Generating High-Performance Tensor Programs for Deep
  Learning,” OSDI 2020, [USENIX paper page](https://www.usenix.org/conference/osdi20/presentation/zheng).
