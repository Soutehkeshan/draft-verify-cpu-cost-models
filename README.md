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

**NANO COLLECTION COMPLETE — MODELS TRAINED AND EVALUATED ON THE SHARED TEST SET**

The Nano re-measured all 5,000 schedules of the M2 pool in one RPC session
from commit `db2cf1c` (TEST first, then VALIDATION, then TRAIN): 4,966 `ok`
and 34 `slow_skip`, no build failures or wrong results, per-batch canary drift
between +0.01% and +0.88%, and no CPU clock drop or throttling. The Nano split
reuses the M2 split, so both targets share the same 752 TEST schedules. The
analytic constants were fitted and XGBoost was trained on TRAIN, with its
settings chosen on VALIDATION, and every scorer was evaluated once on TEST;
see "Nano results" below, `results/jetson_nano/`, and the collection summary
in `data/nano/collection_report.md`. The rest of this section records how the
Nano path was brought up and qualified.

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
| `jetson_nano` | NVIDIA Jetson Nano 4GB Developer Kit; ARM Cortex-A57; four CPU cores; reported Ubuntu 18.04.5, aarch64, L4T R32.5.0 / JetPack 4.5, GCC 7.5.0, Python 3.6.9. CUDA 10.2 is recorded as installed but is not used. | Primary target. Explicit `llvm` schedule rules and postprocessors; SIMD, cache, and register values from the Nano; peak compute and bandwidth fitted on TRAIN. 5,000 schedules measured over RPC. |
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
selecting Apple M2 internally. The M2 profile runs locally (`make smoke`); the
Nano profile cross-compiles on the M2 and runs over RPC, which is how the 5,000
Nano schedules were measured.

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
The Nano pool re-measured the M2 pool's positions, so `02_split.py` reuses the
M2 split named, with its sha256, in `data/nano/run_meta.json`, restricted to
the schedules modelable on the Nano (all 5,000), instead of drawing a new one;
both targets share the same TEST schedules.
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

After the collection:

```sh
make split TARGET=jetson_nano  # data/nano/splits.json from the M2 split (shared TEST set)
make eval TARGET=jetson_nano   # TRAIN-only feature correlations, analytic fit, training,
                               # TEST evaluation, and plots into results/jetson_nano/
```

An existing split is preserved unless `FORCE_SPLIT=1`; the result scripts
require a clean worktree.

## Nano results (TEST, 752 schedules)

`make eval TARGET=jetson_nano` produced these at commit `678e873` with a clean
worktree; the files are in `results/jetson_nano/`. TRAIN (3,499 schedules)
fitted the analytic constants and trained XGBoost, VALIDATION (749) selected
the XGBoost settings (depth 6, learning rate 0.1, 300 trees for both feature
sets), and TEST was used only for the numbers below. XGBoost rows score the
mean prediction of five training seeds. Brackets are 95% bootstrap intervals
over TEST schedules (1,000 resamples); for random ranking they are the
2.5–97.5% range of 1,000 random orders. Latency ratios are lower-is-better.

| Scorer | Spearman | Fastest-decile Spearman | Top-10 recall in 30 | Top-1 latency ratio |
|---|---|---|---|---|
| Random | 0.000 [−0.075, 0.070] | 0.001 | 0.04 [0.00, 0.20] | 6.74 [1.21, 33.57] |
| Analytic, hand-set constants | 0.405 [0.351, 0.462] | −0.004 | 0.10 [0.00, 0.30] | 1.80 [1.38, 5.93] |
| Analytic, TRAIN-fitted constants | 0.442 [0.382, 0.499] | −0.010 | 0.10 [0.00, 0.30] | 1.80 [1.38, 5.93] |
| XGBoost, schedule features | 0.934 [0.921, 0.945] | 0.412 | 0.60 [0.30, 1.00] | 1.31 [1.05, 1.31] |
| XGBoost, schedule + analytical features | 0.940 [0.926, 0.951] | 0.497 | 0.90 [0.60, 1.00] | 1.07 [1.00, 1.17] |

XGBoost ranks the Nano TEST schedules far better than the analytical model
alone, which does not order the fastest decile at all. Adding the analytical
features to XGBoost (`paired_comparison.csv`; metrics averaged over the five
seeds, intervals from 2,000 paired resamples of TEST):

| Metric | Schedule features | + analytical | Difference [95% CI] |
|---|---|---|---|
| Spearman | 0.931 | 0.937 | +0.006 [+0.001, +0.012] |
| Fastest-decile Spearman | 0.400 | 0.472 | +0.073 [−0.026, +0.179] |
| Top-10 recall in 30 | 0.56 | 0.76 | +0.20 [0.00, +0.38] |
| Top-1 latency ratio | 1.250 | 1.096 | −0.155 [−0.229, +0.007] |
| Top-5 latency ratio | 1.187 | 1.122 | −0.065 [−0.156, +0.016] |

The overall Spearman gain is small, its interval excludes zero, and all five
seeds improve (0.927–0.933 to 0.935–0.938). The larger gains near the top of
the ranking point the same way, but their intervals reach zero, so this TEST
set does not establish them; the top-1 ratio improves for four of the five
seeds. By sketch, the Spearman gain holds for sketch 2 (+0.0107 [+0.0015,
+0.0213]), is borderline for sketch 1 (+0.0095 [−0.0002, +0.0197]), and is
absent for sketch 0 (−0.0034 [−0.0130, +0.0055]).

Other outputs:

- **Analytic constants** (`constants.json`): the TRAIN fit moves peak compute
  from 47.3 to 12.3 GFLOP/s (0.26x) and bandwidth from 10.0 to 8.3 GB/s
  (0.83x). These are effective model constants, not measured rates. Fitting
  raises the analytic Spearman from 0.405 to 0.442 and leaves its top-1 ratio
  at 1.80.
- **Analytic penalties** (`ablations.csv`): without the vector penalty the
  fitted analytic model's Spearman falls from 0.442 to 0.136; removing the
  parallel, cache, or register penalty changes it by 0.043 or less, although
  without the cache penalty its top-1 ratio worsens from 1.80 to 5.46.
- **Learning curve** (`learning_curve_summary.csv`; schedule + analytical
  features, mean of five seeds): Spearman 0.796, 0.842, 0.893, 0.906, 0.926,
  and 0.937 with 100, 250, 500, 1,000, 2,000, and 3,499 TRAIN schedules. The
  top-1 ratio does not fall steadily (1.10–1.31).
- **Draft then verify** (`draft_verify.csv`): when XGBoost with schedule
  features drafts 32 candidates and the hybrid model chooses among them, the
  choice reaches the hybrid's own top-1 ratio (1.071). Analytic drafts need 64
  candidates and random drafts 128 to reach it.
- **Timing noise** (`noise_floor.csv`): the Spearman between the medians of
  repeats 1–2 and 3–5 of each TEST schedule is 0.994 overall but 0.579 within
  the fastest decile (75 schedules), so rankings among the fastest schedules
  are limited by timing noise as well as by the models.
- **Leakage check** (`shuffled_label_null.csv`): XGBoost trained on globally
  permuted TRAIN labels has mean TEST Spearman 0.003 (schedule features) and
  0.007 (+ analytical) over 20 permutations each, within two standard errors of
  zero.
- **TRAIN-only feature correlations** (`feature_correlations.csv`):
  vector-lane utilisation (−0.44), parallel utilisation (−0.18), and bytes
  moved (+0.07) correlate with latency in the expected direction; register
  utilisation is flat (−0.001) and is used only by the analytic penalty.
- **Cross-target ranking:** on the 4,966 schedules that were `ok` on the Nano,
  Nano and M2 latencies have Spearman 0.861 (`data/nano/collection_report.md`).

Figures: `results/jetson_nano/plots/overview.pdf` and
`results/jetson_nano/plots/measured_vs_predicted.pdf`.

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
peak check. The collection then measured all 5,000 M2-pool schedules; all
passed the correctness check, and 34 exceeded the slow threshold and kept a
single timing (`slow_skip`).

**Existing M2 evidence:** the released M2 dataset and historical M2 result
artifacts. These are not re-labeled or reinterpreted as Nano measurements.

**Not covered:** positions outside the 5,000 replayed M2-pool schedules (no
new positions were sampled on the Nano), other workloads and data types,
hardware counters, and pinned CPU or memory clocks. The qualification timings
remain diagnostic only; the Nano results come from the collection.

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
   peak compute and bandwidth there are initial guesses; the TRAIN-only fit
   is in `results/jetson_nano/constants.json`.
4. Continue using the isolated Python 3.10 RPC environment; do not modify
   system Python, CUDA, JetPack, drivers, or system packages.
5. Done: the Nano collection (`data/nano/`), the split derived from the M2
   split, training, and the TEST evaluation (`results/jetson_nano/`; see
   "Nano results"). Rerun with `make split TARGET=jetson_nano` and
   `make eval TARGET=jetson_nano` from a clean worktree.

## Limitations and references

The RPC path uses the repository's existing TVM MetaSchedule implementation
and version. The Nano measured 5,000 of the 3,704,400 encoded positions (the
M2 pool's uniform sample) plus 48 bounded qualification candidates. CPU clocks
are not pinned (`schedutil` governor; `jetson_clocks` and the memory clock need
root and were not read), and some schedules shift by a few percent between RPC
sessions, so the collection ran in a single session. TEST was measured first
and TRAIN last within one 2 h 41 min run; canary drift stayed below 1% in every
batch. Analytical features are
estimates and omit hardware counters. The current workload is only one FP32
matrix size. Timing quality depends on device power, thermal, and scheduling
conditions.

- Qiao et al., “Pruner,” ASPLOS 2025.
- Zheng et al., “Ansor: Generating High-Performance Tensor Programs for Deep
  Learning,” OSDI 2020, [USENIX paper page](https://www.usenix.org/conference/osdi20/presentation/zheng).
