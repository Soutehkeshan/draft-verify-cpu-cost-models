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

**NANO RPC QUALIFICATION COMPLETE — NANO COLLECTION ENABLED (EXPLICIT `llvm` RULES)**

The M2 successfully cross-compiled and exported AArch64 CPU code for
`aarch64-linux-gnu` / `cortex-a57`, then loaded and executed it on the Nano
through TVM RPC. The vector result matched NumPy. One 512x512 FP32 matmul
candidate from the existing MetaSchedule path also matched NumPy and returned
a remote `time_evaluator` result. These are integration checks, not a full
schedule collection or model evaluation.

The host-side qualification explicitly used TVM's `llvm` schedule rules and
postprocessors for the Cortex-A57 target. It compiled, linked, executed, and
checked 48 deterministic candidates across all three generated sketches (16
per sketch); all 48 passed the harness's NumPy correctness check. Three
separate seven-repeat timing batches were also run for each sketch's recorded
default schedule. Batch medians were consistent, but two of the three sketches
had isolated repeats more than 10% above their combined median. That bounded
check captured no frequency or temperature data; the later noise canary
(see "Nano qualification") records both before and after every timed round.

This is a small validation sample, not proof that all 3,704,400 encoded
positions are valid or reliable. A broader qualification (layout parity, loop
order, a 150-schedule stratified replay pilot, and a 20-schedule noise canary)
is described under "Nano qualification" below; its outputs are in
`data/nano/`. The Nano profile now sets its schedule rules and postprocessors
explicitly to `llvm` (never `from-target`, which would pick ARM-specific rules
for this target), which opens the Nano collection path. SIMD width, cache sizes, and register count are set from
the Nano's own sysfs, `lscpu`, and ISA information; peak compute and memory
bandwidth are initial guesses that the analytical fit refines on TRAIN only.

The prior Apple M2 data release and its results are retained as historical,
M2-only material. They are not Jetson measurements and are not presented as
evidence for the Nano research question.

## Scope and hardware profiles

Profiles are stored in [`configs/hardware/`](./configs/hardware/), separate
from the model implementations.

| Profile | Known configuration | Status |
|---|---|---|
| `jetson_nano` | NVIDIA Jetson Nano 4GB Developer Kit; ARM Cortex-A57; four CPU cores; reported Ubuntu 18.04.5, aarch64, L4T R32.5.0 / JetPack 4.5, GCC 7.5.0, Python 3.6.9. CUDA 10.2 is recorded as installed but is not used. | Primary target; compiler target verified by the RPC proof of concept. Schedule rules and model assumptions remain unset. |
| `apple_m2` | MacBook Air Apple M2, arm64, four cores; existing TVM LLVM target and M2 study assumptions. | Compatibility profile for the previously collected M2 dataset and local smoke tests. |

The M2 profile's 400 GFLOP/s and 100 GB/s values are legacy analytical-fit
starting assumptions, not measured hardware peaks. Its cache and vector
assumptions are retained only to reproduce the earlier M2 work. Nano peak
compute and bandwidth must be calibrated from TRAIN measurements (or supplied
with a documented source); unknown Nano values are never filled with M2
defaults.

The host uses its existing TVM 0.27.0.post1 installation and LLVM 22 for
cross-compilation. The Nano-side RPC server runs from its isolated Python 3.10
TVM wheel environment; the system Python and NVIDIA stack are not part of this
workflow.

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
- `src/dvcm/rpc.py`: TVM RPC session, module export/upload/link timing, and
  remote inputs.
- `src/dvcm/device_state.py` and `scripts/nano_device_state.py`: read-only Nano
  clock, frequency-residency, thermal, power-mode, and load snapshots.
- `src/dvcm/telemetry.py` and `scripts/nano_telemetry.py`: the Nano thread
  sampler used by the threading and sampler-overhead checks.
- `scripts/`: collection, splitting, fitting, training, evaluation, and plots.
- `scripts/validate_nano_schedules.py`: bounded RPC schedule qualification;
  prints validation metadata and results without writing a dataset.
- `scripts/qualify_nano.py`: Nano qualification steps (see below);
  `scripts/nano_fma_peak.c`: NEON FMA throughput check.
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
new results to `results/apple_m2/`; the Nano profile reserves `data/nano/`
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
target. Sampling new positions (`make data`) is reserved for the M2 target;
for the Nano, `01_collect.py` stops with an error that points to the
`--positions-from` replay command under "Nano qualification". To rerun the two
integration checks through an SSH tunnel, use
`python scripts/rpc_poc.py --host 127.0.0.1 --port 9090`. This runs one vector
function and one MetaSchedule matmul candidate; it does not collect schedules.
To repeat the bounded schedule qualification and print its per-candidate
results as JSON without writing a dataset, use:

```sh
TVM_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python scripts/validate_nano_schedules.py --host 127.0.0.1 --port 9090
```

TVM's CPU pool is configured by the **Nano RPC server process**, not by the M2
client. In TVM 0.27, `TVM_NUM_THREADS` takes precedence over
`OMP_NUM_THREADS`; `runtime.NumThreads()` reports the effective pool size.
`TVM_BIND_THREADS=1` enables the runtime's worker affinity behavior,
`TVM_EXCLUDE_WORKER0=1` explicitly uses the calling thread as worker zero, and
`TVM_THREAD_POOL_SPIN_COUNT=0` avoids long idle spinning. These are per-process
environment settings and do not change Nano system settings.

Use an isolated test port, leaving the normal port 9090 server untouched.
Change only `TVM_NUM_THREADS` and `OMP_NUM_THREADS` to `1` for the
single-thread run, restart the temporary server, then repeat with both set to
`4`; keep the other three settings unchanged:

```sh
env CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  TVM_NUM_THREADS=4 OMP_NUM_THREADS=4 TVM_BIND_THREADS=1 \
  TVM_EXCLUDE_WORKER0=1 TVM_THREAD_POOL_SPIN_COUNT=0 \
  "$HOME/.local/envs/tvm-rpc/bin/python" \
  -m tvm.exec.rpc_server --host 127.0.0.1 --port 9091
```

On the M2, forward that Nano port to local port 19091:

```sh
ssh -N -L 19091:127.0.0.1:9091 <nano-ssh-host>
```

For each configuration, run the bounded parallel-vector test:

```sh
PYTHONPATH=src python scripts/validate_nano_threading.py \
  --ssh-target arad@jetson-nano.local --host 127.0.0.1 --port 19091 \
  --device-rpc-port 9091 --expected-threads 4
```

The test verifies the environment on the Nano session process, checks
`runtime.NumThreads()`, compiles a TIR-parallel `y = 2*x + 1` function for the
Nano target, and checks its NumPy result. It reports an uninstrumented timing
separately from a dedicated instrumented worker-attribution pass. Four-thread
success requires evidence of three active TVM pool workers and activity on all
four CPU IDs; thread counts alone are not treated as worker-activity evidence.
No schedules or dataset rows are collected.

Measure the two sampler modes while the server is idle (not during the clean
timings):

```sh
PYTHONPATH=src python scripts/measure_nano_sampler_overhead.py \
  --ssh-target arad@jetson-nano.local --rpc-host 127.0.0.1 \
  --rpc-port 19091 --device-rpc-port 9091
```

It reports sampler process CPU time and per-sample cost for thread-only polling
and for the full sampler, which additionally reads system utilization and
hardware sensors. The latter is not assumed to be non-intrusive.

The older three-schedule timing sampler can still be run with
`scripts/measure_nano_telemetry.py`, but do not use it for a performance
comparison until the independent overhead result has been reviewed. The
sampler follows the TVM RPC process tree to the per-connection session child;
if multiple clients share the selected server, pass `--rpc-pid` for that
session process. `--device-rpc-port` identifies the Nano-side server port when
the local tunnel uses a different port.

The temporary server uses `/usr/bin/g++` for TVM's target-side archive linking
and listens only on localhost. Keep the SSH tunnel open while running tests.
The RPC server permits remote code execution and file writes; do not expose it
on an untrusted network.

## Nano qualification

`scripts/qualify_nano.py` runs the qualification steps without collecting a
dataset. It only reads the M2 data in `data/` and writes to `data/nano/`
(`run_meta.json`, `pilot_report.json`, `canary.json`, `layout_*.json`, and
`qualification/`). The RPC server runs with the fixed configuration recorded
in `data/nano/run_meta.json`. No telemetry sampler runs during timing; the
read-only `scripts/nano_device_state.py` probe records clocks, frequency
residency, temperatures, cooling states, `nvpmodel`, and load before and after
each timed block.

```sh
PYTHONPATH=src python scripts/qualify_nano.py parity        # layout/design-space parity vs M2
PYTHONPATH=src python scripts/qualify_nano.py loop-order    # SSRSRS order in scheduled TIR
PYTHONPATH=src python scripts/qualify_nano.py replay --ssh-target arad@jetson-nano.local
PYTHONPATH=src python scripts/qualify_nano.py noise --ssh-target arad@jetson-nano.local
PYTHONPATH=src python scripts/qualify_nano.py throughput
PYTHONPATH=src python scripts/qualify_nano.py features
```

`replay` rebuilds 150 M2-pool schedules (stratified by sketch and M2 latency
decile, fixed seed, no resampling) on the Nano target, checks NumPy
correctness, times them with the M2 protocol, and records
`slow_threshold_ms` (20 x pilot median) in `data/nano/run_meta.json`.

The Nano collection path re-measures the M2
pool's positions in `test_first` order: TEST ids from `data/splits.json`, then
VALIDATION, then TRAIN in nested-subset order (100, 250, 500, 1000, 2000,
all). It checkpoints per batch (`--batch-size`, default 250; `--cooldown-s`,
default 60), resumes, retries once after an RPC drop (`rpc_error` otherwise),
and refuses to run without the pilot threshold, the pilot report, the
qualified canaries, a clean worktree (or `--allow-dirty`), or an open
collection gate. It pauses (exit code 3, reason in `run_meta.json`) when the
canary median drifts more than 5% from the qualification baseline in two
consecutive batches, when the time at the top CPU clock does not cover the
batch's kernel-busy time or a CPU cap/throttle device activates, when more
than 5% of a batch's rows are not `ok`, after two RPC reconnects in a row, or
when the RPC server environment no longer matches `run_meta.json`:

```sh
PYTHONPATH=src python scripts/01_collect.py --target jetson_nano \
  --rpc-host 127.0.0.1 --rpc-port 9090 --positions-from data/pool.parquet \
  --order test_first --ssh-target arad@jetson-nano.local [--n 752]
```

`python scripts/report_nano_collection.py` (safe to run at any time, also
during collection) writes `data/nano/collection_report.md`: rows by status,
RPC session indices, per-batch canary drift, frequency and temperature,
throttling, M2-vs-Nano Spearman (pooled and per sketch), the Nano latency
distribution, and output checksums.

## What is tested and what remains

**Tested locally:** profile loading and validation; target-bound feature
requirements; analytic estimation and fitting on synthetic test inputs;
XGBoost training on synthetic test inputs; deterministic split validation;
ranking metrics; TVM M2 schedule generation, schedule postprocessing, compile,
correctness, and smoke timing. No synthetic measurements are used as
experimental results.

**Tested over Nano RPC:** AArch64 vector compilation/export/link/load/CPU
execution; 48 candidates across all three 512x512 matmul sketches generated
with TVM MetaSchedule `llvm` rules and postprocessors; NumPy correctness for
all sampled schedules. Three seven-repeat timing batches were run for one
representative schedule per sketch. Batch medians were consistent, with
isolated high repeats on two sketches. macOS archive export must suppress
AppleDouble metadata (`COPYFILE_DISABLE=1`) so the Nano linker receives only
the object files. The optional collector keeps schedule build, archive export,
module upload, remote link/load, and kernel timing distinct; initial input
generation and transfer are recorded once in run metadata. The qualification
added: design-space and loop-order parity with the M2 collection; 150
replayed M2-pool schedules, all correct and timed with the M2 protocol; a
20-schedule canary over five rounds and a second RPC session; and a NEON FMA
peak check.

**Existing M2 evidence:** the released M2 dataset and historical M2 result
artifacts. These are not re-labeled or reinterpreted as Nano measurements.

**Awaiting:** schedule-space reliability beyond the 198 qualified positions,
TRAIN-calibrated compute/bandwidth constants, and the Nano collection,
training, and TEST evaluation. No Nano dataset or model-evaluation results are
reported; the qualification timings are diagnostic only.

## Handoff: Nano-specific work

1. The Nano OS, kernel, L4T, GCC, Python, and TVM versions and the fixed RPC
   server command and environment are recorded in `data/nano/run_meta.json`.
   The Nano does not need an LLVM compiler for the cross-compilation workflow.
2. The target `aarch64-linux-gnu` / `cortex-a57` with explicit `llvm` schedule
   rules and postprocessors has design-space and loop-order parity with the M2
   collection and passed the 150-schedule replay pilot and noise canary, so
   `configs/hardware/jetson_nano.json` now sets both fields to `llvm`.
3. SIMD width, cache sizes, and register count are set in
   `configs/hardware/jetson_nano.json` with their sources in its `notes`;
   peak compute and bandwidth there are initial guesses, fitted on TRAIN only.
4. Continue using the isolated Python 3.10 RPC environment; do not modify
   system Python, CUDA, JetPack, drivers, or system packages.
5. Collect with the Nano collection path (see "Nano qualification"), then
   split, train, and evaluate on the shared TEST set.

## Limitations and references

The RPC path uses the repository's existing TVM MetaSchedule implementation
and version. Only 198 of the 3,704,400 encoded positions (48 bounded
candidates and 150 replayed M2-pool schedules) have been validated on the
Nano before collection. CPU clocks are not pinned (`schedutil` governor;
`jetson_clocks` and the memory clock need root and were not read), and some
schedules shift by a few percent between RPC sessions. Analytical features are
estimates and omit hardware counters. The current workload is only one FP32
matrix size. Timing quality depends on device power, thermal, and scheduling
conditions.

- Qiao et al., “Pruner,” ASPLOS 2025.
- Zheng et al., “Ansor: Generating High-Performance Tensor Programs for Deep
  Learning,” OSDI 2020, [USENIX paper page](https://www.usenix.org/conference/osdi20/presentation/zheng).
