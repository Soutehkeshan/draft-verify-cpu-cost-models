# CPU Cost Models on a Jetson Nano

## Research question

**How well do analytical and machine-learning cost models rank CPU schedules
on an NVIDIA Jetson Nano, and do analytical features improve XGBoost's
ranking?**

The study uses one workload: FP32 matrix multiplication with `M=N=K=512`. It
compares a Pruner-style analytical model, XGBoost on schedule features, and
XGBoost on schedule plus analytical features ("the hybrid"), with random
ranking as the baseline. The project is CPU-only; CUDA and GPU scheduling are
out of scope. A Mac laptop compiles every schedule with TVM and sends it to
the Nano over TVM RPC; all timings are taken on the Nano.

## Main results (TEST, 752 schedules)

| Scorer | Spearman | Fastest-decile Spearman | Top-10 recall in 30 | Top-5 latency ratio |
|---|---|---|---|---|
| Random | 0.000 [−0.075, 0.070] | 0.001 | 0.04 | 6.855 |
| Analytic, hand-set constants | 0.405 [0.351, 0.462] | −0.004 | 0.10 | 2.443 |
| Analytic, TRAIN-fitted constants | 0.442 [0.382, 0.499] | −0.010 | 0.10 | 2.443 |
| XGBoost, schedule features | 0.931 [0.917, 0.941] | 0.400 | 0.56 | 1.187 |
| XGBoost, schedule + analytical | 0.937 [0.923, 0.948] | 0.472 | 0.76 | 1.122 |

XGBoost rows are means over five training seeds with 95% bootstrap
intervals (2,000 TEST resamples); analytic rows score the whole TEST set
(1,000 resamples); the random row is the mean of 1,000 random orders. Latency
ratios are lower-is-better. In short:

- XGBoost ranks the Nano schedules far better than the analytical model,
  which does not order the fastest decile at all.
- Adding the analytical features raises Spearman by **+0.006 [+0.001,
  +0.012]** (seed-paired); all five seeds improve. Gains near the top of the
  ranking point the same way (top-10 recall +0.20 [0.00, +0.38], top-1 ratio
  −0.155 [−0.229, +0.007]) but their intervals reach zero.
- The features help only from about 500 TRAIN schedules on.
- **v3.0 confirmation on 20,000 new schedules:** the overall gain holds
  (+0.005 [+0.004, +0.006]), but the gains near the top do not; see
  "Confirmation results" below.
- The vector penalty carries the analytical model, which sorts schedules into
  a few coarse groups; as a drafter it keeps few of the fastest schedules.
- A second timing of TEST agrees at Spearman 0.998 overall and 0.953 within
  the fastest decile, so the models, not timing noise, limit the results.

Details for each point are in "Results" below.

## Repository layout

```
configs/hardware/jetson_nano.json   Nano hardware values, compiler target, data/result folders
configs/workloads/matmul_fp32_512.json   matrix size, dtype, seed, timing repeats
src/dvcm/                           shared library
scripts/                            the programs (numbered = main pipeline)
tests/                              unit tests (no Nano needed)
data/jetson_nano/                   the measured dataset (not in Git)
results/jetson_nano/                everything computed from it (not in Git)
Report/                             the LaTeX report and its generated tables and figures (not in Git)
```

**Library (`src/dvcm/`)**

| Module | Job |
|---|---|
| `config.py` | Load and check the hardware and workload profiles. |
| `encoding.py` | Read a schedule's tile and unroll choices from a TVM trace as 11 integers, check them, and write them back into a trace. |
| `harness.py` | Build the matmul, apply a schedule, compile it for the Nano, check the result and time it over RPC. |
| `rpc.py` | Upload, load and run compiled modules on the Nano. |
| `device_state.py` | Read-only Nano clock, temperature and load snapshots between timing runs. |
| `features.py` | Schedule features and derived analytical features. |
| `analytic.py` | The analytical latency model and its TRAIN-only constant fit. |
| `learned.py` | XGBoost on log latency; the 8-setting search grid. |
| `metrics.py` | Spearman, fastest-decile Spearman, top-10 recall in 30, top-k latency ratios. |
| `analysis.py` | Bootstrap resampling, seed-paired differences, Spearman-Brown, split halves. |
| `study.py` | Load the pool, layouts and split; validate the split. |
| `provenance.py` | Record the Git commit and inputs next to every output; refuse uncommitted code. |

**Scripts (`scripts/`)**

| Script | Job | Writes |
|---|---|---|
| `01_collect.py` | Time a pool's schedules on the Nano over RPC, in batches, with canaries and auto-pause | `data/jetson_nano/` |
| `nano_device_state.py` | Read-only probe run on the Nano by `device_state.py` | — |
| `retime_nano_test.py` | Second timing of the TEST schedules in a new RPC session | `data/jetson_nano/test_retime*` |
| `check_feature_correlations.py` | TRAIN-only correlation of each feature with latency | `feature_correlations.csv` |
| `02_fit_analytic.py` | Fit the analytical model's two constants on TRAIN | `constants.json` |
| `03_train.py` | Choose XGBoost settings on VALIDATION; train 5 seeds × 6 TRAIN sizes × 2 feature sets | `training_grid.json`, `models/` |
| `04_evaluate.py` | Score everything on TEST | `evaluation.csv` and the files listed below |
| `check_shuffled_label_null.py` | Leakage check: train on shuffled labels, 120 times | `shuffled_label_null_extended*.csv` |
| `make_report_tables.py`, `make_report_figures.py` | Report tables and figures from the result files | `Report/tables/`, `Report/figures/` |
| `check_report_numbers.py` | Check every number in `Report/*.tex` against the result files | — |
| `nano_fma_peak.c` | NEON FMA peak test, the source of the hand-set 47.3 GFLOP/s | — |

## Data

`data/jetson_nano/` (to be published as a release; see the report's
"Code and data" line):

| File | Holds |
|---|---|
| `pool.parquet` | One row per schedule: `id`, `sketch_id`, `vector` (11 encoded choices), `repeats_ms`, `median_ms`, `std_ms`, `status`, build/upload/load times, batch, timestamp, device state |
| `splits.json` | TRAIN (3,499) / VALIDATION (749) / TEST (752) ids and nested TRAIN subsets (100, 250, 500, 1,000, 2,000, all) |
| `layout_0.json` … `layout_2.json` | How to read the 11 numbers for each of the three sketches |
| `run_meta.json` | Nano software, RPC server command and environment, slow threshold, pause rules, status counts |
| `canary.json`, `canary_log.parquet`, `collection_batches.jsonl`, `collection.log` | Canary schedules, every canary timing, per-batch drift, clock and temperature |
| `collection_report.md` | A readable summary of the collection |
| `test_retime.parquet` (+ `_canary`, `_batches`, `_meta`) | The second timing of TEST |
| `pilot_report.json`, `qualification/` | The bring-up checks run before the collection (see below) |

The 5,000 schedules are a uniform random sample of the 3,704,400 valid
encoded schedules; they and the stratified 70/15/15 split were drawn once,
for an earlier timing of the same schedules on another machine, and are
reused unchanged. `splits.json` records the source split's path and sha256
(`derived_from`), and `run_meta.json` records the source pool's. The split is
part of the dataset; nothing regenerates it.

**Collection.** All 5,000 schedules were timed in one RPC session (2 h 41 min,
TEST first, then VALIDATION, then TRAIN): 4,966 `ok` and 34 `slow_skip`
(slower than 718 ms, timed once), no build failures or wrong results. Each
schedule ran five times on four threads; its latency is the median. Twenty
canary schedules re-timed after every 250-schedule batch drifted between
+0.01% and +0.88%, and the CPU clock never dropped or throttled. The
collector pauses itself when canaries drift more than 5% in two consecutive
batches, the clock drops, a cap or throttle activates, more than 5% of a
batch fails, two RPC reconnects happen in a row, or the RPC server's
environment changes.

**Bring-up checks** (before the collection; scripts removed on 2026-11-18,
in Git history at `51e2a4d`; outputs kept in `data/jetson_nano/`): the
schedule space and loop order of the cross-compiled code; a 150-schedule
replay pilot, all correct, which set the slow threshold to 20 times the pilot
median; a 20-schedule noise canary over five rounds and two RPC sessions;
the thread-pool configuration (`qualification/threading/`); and the NEON FMA
peak (46.5 GFLOP/s on four cores at 1,479 MHz).

**Re-timing.** `retime_nano_test.py` timed all 752 TEST schedules again with
15 repeats each, the slow-skip shortcut off, in a seeded random order, in one
new RPC session to the same server process. All 752 were `ok`; canary drift
stayed between +0.35% and +0.62%.

## Method

- **Analytical model** (`analytic.py`): latency = FLOPs / (P × vector ×
  parallel × register) + bytes moved / (W × L1 × L2), each penalty in [0, 1].
  P and W start at 47.3 GFLOP/s and 10.0 GB/s (hardware tests) and are fitted
  on TRAIN to 12.3 GFLOP/s and 8.3 GB/s; they are effective constants, not
  measured rates. Cache sizes, lanes, cores and registers come from the Nano
  profile.
- **XGBoost** (`learned.py`): predicts log latency from the 11 encoded choices
  and the sketch. The hybrid adds two tile working sets, bytes moved,
  vector-lane and parallel utilisation, and the analytical prediction. Each
  feature set chose its depth, learning rate and number of trees from eight
  settings on VALIDATION; both chose depth 6, rate 0.1, 300 trees.
- **Seeds.** Each model is trained with seeds 0–4 (each tree sees a random 80%
  of rows and columns). Table rows are means over seeds; `evaluation.csv`
  scores the mean prediction of the five seeds (an ensemble), so its XGBoost
  rows differ slightly (0.934 and 0.940).
- **Learning curve.** Both models train on each nested TRAIN subset; the
  hybrid's analytical constants are refit on that subset alone.
- **Metrics** (`metrics.py`): Spearman; Spearman within the measured-fastest
  10% (75 schedules); top-10 recall in 30; top-1/5/10 latency ratio (mean
  measured latency of the model's top k over the best; 1 is perfect).
- **Uncertainty.** 95% bootstrap intervals over TEST schedules. Differences
  between the two XGBoost models are paired by seed and use 2,000 paired
  resamples (seed 20260 + scope index), the same for every file.
- **Discipline.** Constants fitted on TRAIN only, settings chosen on
  VALIDATION only, TEST used only for reporting. Result scripts refuse to run
  on uncommitted code and record the commit and input checksums in a
  `.meta.json` next to every output.

## Results

All files are in `results/jetson_nano/`, each with a `.meta.json` recording
its provenance. They were produced at `eee2a4c` (the 120-permutation leakage
check at `eba8e42`). On 2026-11-18 the trimmed code reproduced every one of
them exactly, from a fresh committed copy (correlations, fit, training,
evaluation, leakage).

**Adding the analytical features** (`paired_comparison.csv`, mean of five
seed-paired differences):

| Metric | Schedule features | + analytical | Difference [95% CI] |
|---|---|---|---|
| Spearman | 0.931 | 0.937 | +0.006 [+0.001, +0.012] |
| Fastest-decile Spearman | 0.400 | 0.472 | +0.073 [−0.026, +0.179] |
| Top-10 recall in 30 | 0.56 | 0.76 | +0.20 [0.00, +0.38] |
| Top-1 latency ratio | 1.250 | 1.096 | −0.155 [−0.229, +0.007] |
| Top-5 latency ratio | 1.187 | 1.122 | −0.065 [−0.156, +0.016] |

All five seeds improve Spearman (0.927–0.933 to 0.935–0.938); the top-1 ratio
improves for four of five. By sketch, the Spearman gain holds for sketch 2
(+0.0107 [+0.0015, +0.0213]), is borderline for sketch 1 (+0.0095 [−0.0002,
+0.0197]) and absent for sketch 0 (−0.0034 [−0.0130, +0.0055]).

**Learning curve** (`learning_curve_summary.csv`, `learning_curve_paired.csv`;
Spearman, mean of five seeds):

| TRAIN schedules | Schedule features | + analytical | Difference [95% CI] |
|---|---|---|---|
| 100 | 0.798 | 0.791 | −0.007 [−0.020, +0.008] |
| 250 | 0.837 | 0.840 | +0.004 [−0.006, +0.014] |
| 500 | 0.880 | 0.891 | +0.011 [+0.003, +0.019] |
| 1,000 | 0.895 | 0.906 | +0.011 [+0.004, +0.019] |
| 2,000 | 0.916 | 0.926 | +0.010 [+0.005, +0.016] |
| 3,499 | 0.931 | 0.937 | +0.006 [+0.001, +0.012] |

Near the top of the ranking there are 24 differences (fastest-decile
Spearman, top-10 recall, top-1 and top-5 ratios at six sizes); chance alone
would put about one outside zero. Two are: the top-1 ratio with 100 schedules
(−0.322 [−0.574, −0.084]) and the top-5 ratio with 2,000 (−0.162 [−0.264,
−0.031]), both favouring the hybrid, which is what chance gives.

**Analytical model diagnosis**

- *Penalties* (`ablations.csv`): without the vector penalty the fitted
  model's Spearman falls from 0.442 to 0.136; removing the parallel, cache or
  register penalty changes it little, although without the cache penalty its
  top-1 ratio worsens from 1.80 to 5.46.
- *Compute versus memory* (`analytic_time_shares.csv`): compute time is 94.8%
  of the median schedule's predicted latency; memory time dominates for 2.7%
  of schedules.
- *Penalty values* (`analytic_penalties.csv`): the vector factor is 0.25 for
  220 TEST schedules, 0.5 for 187 and 1 for 345; the parallel factor is 1 for
  717 and the register factor 1 for 691. The L1 factor is below 1 for 143
  schedules and the L2 factor 0.667 for 40.
- *Cache penalties* (`analytic_cache_effect.csv`): they reorder 2.5% of TEST
  pairs, but change the predicted fastest schedule and keep only 2 of the
  predicted top 10.
- *Feature correlations* (`feature_correlations.csv`, TRAIN only): vector-lane
  utilisation −0.44, parallel utilisation −0.18, bytes moved +0.07, register
  utilisation −0.001.

**Draft then verify** (`draft_verify.csv`): the analytical model's top 32
holds one of the ten fastest TEST schedules (nine in its top 256); XGBoost's
top 32 holds six. When XGBoost with schedule features drafts 32 candidates and
the hybrid chooses among them, the choice reaches the hybrid's own top-1
ratio (1.071); analytical drafts need 64 candidates and random drafts 128.

**Noise ceiling and re-timing** (`noise_floor.csv`, `retime_agreement.csv`,
`retime_evaluation.csv`): splitting each schedule's repeats into odd- and
even-numbered halves gives a Spearman-Brown reliability of 0.999 overall and
0.945 in the fastest decile (the first-two-versus-last-three split is
confounded by warm-up: the first repeats are slower for 62% of schedules).
The second timing agrees with the first at Spearman 0.998 overall and 0.953
within the fastest decile (0.950, 0.972 and 0.933 per sketch), although 21%
of schedules moved by more than 5%. Against the re-timed medians every scorer
keeps its place; the hybrid scores Spearman 0.939, fastest-decile 0.533 and
top-1 ratio 1.065. The models' fastest-decile Spearman (0.400–0.472) is far
below the ~0.95 the measurements allow.

**Leakage check** (`shuffled_label_null_extended_summary.csv`): over 120 global
permutations of the TRAIN labels, mean TEST Spearman is 0.006 (schedule
features) and 0.009 (+ analytical); fastest-decile Spearman −0.013 and
−0.026. The largest, −2.25 standard errors, is negative, which is not the
direction a leak would push it.

## Confirmation study (v3.0, planned before any new timing)

The original TEST set (752 schedules, so only ten "fastest ten") left the
gains near the top of the ranking undecided. To settle them, 20,000 new
schedules, none of them in the original pool, are drawn uniformly with seed 1
by `scripts/sample_confirmation_pool.py`, timed on the Nano with the original
protocol into `data/jetson_nano_confirm/`, and scored by
`scripts/evaluate_confirmation.py` (`make confirm`). The models, settings and
analytic constants are the ones fitted on the original TRAIN set; nothing is
retrained or re-tuned.

Fixed before the new timings: four primary gains from adding the analytical
features at full TRAIN, namely overall Spearman, fastest-decile Spearman,
top-1% recall in the model's top 3%, and the top-1 latency ratio. Each counts
as confirmed only if its 95% paired range (2,000 resamples) lies on the better
side of zero, and all four are reported whatever they show. Secondary: the
gain at every TRAIN size, the draft test with the fastest 1% as targets, and
the repeat-noise ceiling of the new timings. The schedules are timed in a
random order, so if the run stops early, the ones timed so far are still a
random sample and are analysed as they are.

### Confirmation results (v3.0)

All 20,000 schedules were timed (19,854 fully, 146 once because they ran
over the slow threshold). Canary drift stayed between +0.9% and +3.8%, under
the 5% pause limit, and the clock never dropped. The run took two RPC
sessions: the Nano's RPC worker keeps every module it loads mapped in memory,
about four mappings each, and after 15,059 schedules it reached Linux's
per-process limit (`vm.max_map_count` = 65,530), so the remaining 4,941 were
timed in a fresh session. A third session, cut short by a network drop after
941 schedules, was deleted before any analysis and its schedules were timed
again in the second session (`data/jetson_nano_confirm/sessions_note.json`).

Gain from adding the analytical features (hybrid minus plain XGBoost, full
TRAIN, five seed pairs, 95% paired range from 2,000 resamples):

| Metric | Original TEST (752) | Confirmation (20,000) | Confirmed |
|---|---|---|---|
| Spearman | +0.006 [+0.001, +0.012] | +0.005 [+0.004, +0.006] | yes |
| Fastest-decile Spearman | +0.073 [−0.026, +0.179] | −0.001 [−0.024, +0.019] | no |
| Fastest found (top-10 in 30 / top-1% in 3%) | +0.20 [0.00, +0.38] | +0.03 [−0.02, +0.08] | no |
| Top-1 latency ratio | −0.155 [−0.229, +0.007] | −0.080 [−0.164, +0.466] | no |
| Top-5 latency ratio (not primary) | −0.065 [−0.156, +0.016] | +0.146 [−0.041, +0.325] | no |

- **The overall gain is real and now precise.** +0.005 against +0.006
  before, with a range ten times narrower; all five seeds improve, and the
  gain holds in each of the three sketches separately.
- **The larger gains near the top did not hold.** With about 200 schedules
  in the fastest 1% instead of ten, the hybrid finds the best schedules no
  better than plain XGBoost: the fastest-decile gain is zero and the other
  top-of-ranking ranges all include zero. The bigger numbers on the original
  TEST set were luck from a small sample.
- **The gain shrinks as XGBoost gets more data.** Overall Spearman gains
  +0.001 at 100 TRAIN schedules (not significant), then +0.013, +0.017,
  +0.012, +0.007 and +0.005 at 250, 500, 1,000, 2,000 and 3,499, all
  significant. Near the top, the hybrid helps only with little data
  (fastest-decile gain +0.042 at 250 and +0.026 at 500; top-1% recall +0.05
  and +0.06 at 100 and 250), and the help is gone by 1,000. Four of the
  15 top-of-ranking comparisons below full TRAIN exclude zero, against about
  one expected by chance; as secondary results they are a lead, not a
  finding.
- **The formula alone stays weak, and drafts poorly.** Spearman 0.471
  (fitted constants), 0.127 within the fastest decile. Of the fastest 200
  schedules, its top 200 holds 5 and its top 800 holds 24; XGBoost's hold 69
  and 135.
- **The timings are reliable.** Splitting each schedule's five repeats in
  two, the halves agree at Spearman 0.996 overall and 0.829 within the
  fastest decile, while the models reach 0.43 there: the models, not the
  measurements, are what limit the ranking of the best schedules.

Results with provenance are in `results/jetson_nano_confirm/` (`summary.json`,
`paired.csv`, `scores.csv`, `learning_curve.csv`, `draft.csv`, `noise.csv`),
produced by `make confirm` at commit `d4e814d`, the commit that fixed the plan.

## Running it

Use the pinned dependencies on the development machine (not on the Nano):

```sh
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
make test          # unit tests, no Nano needed
make eval          # feature check, analytic fit, training, TEST evaluation -> results/jetson_nano/
make leakage       # 120-permutation shuffled-label check (slow)
make report        # report tables and figures, then check every number in Report/*.tex
```

Result scripts require a clean, committed worktree; `ALLOW_DIRTY=1` overrides
it and records the dirty state in provenance.

**Timing on the Nano.** The RPC server runs on the Nano from an isolated
Python 3.10 TVM environment with fixed thread settings (recorded in
`data/jetson_nano/run_meta.json`):

```sh
env CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  TVM_NUM_THREADS=4 OMP_NUM_THREADS=4 TVM_BIND_THREADS=1 \
  TVM_EXCLUDE_WORKER0=1 TVM_THREAD_POOL_SPIN_COUNT=0 \
  "$HOME/.local/envs/tvm-rpc/bin/python" \
  -m tvm.exec.rpc_server --host 127.0.0.1 --port 9090
```

The server listens only on the Nano's localhost; forward it from the Mac with
`ssh -N -L 9090:127.0.0.1:9090 <nano-ssh-host>`. It allows remote code
execution, so never expose it on an untrusted network. To time a pool's
schedules (for example a copy of the released dataset) into a fresh output
folder that holds the qualification files (`run_meta.json` with the slow
threshold, `pilot_report.json`, `canary.json`):

```sh
PYTHONPATH=src python scripts/01_collect.py --positions-from <pool.parquet> \
  --output-dir <folder> --rpc-host 127.0.0.1 --ssh-target <nano-ssh-host>
PYTHONPATH=src python scripts/retime_nano_test.py --rpc-host 127.0.0.1 \
  --ssh-target <nano-ssh-host>   # second timing of TEST, 15 repeats each
```

## Limitations

One FP32 matrix size on one device. The 5,000 schedules are a uniform sample
of 3,704,400, while a real search concentrates on good candidates. The TEST
set holds only ten "fastest ten" schedules, so top-of-ranking intervals are
wide, and one TEST set serves every analysis; the 20,000-schedule
confirmation set addresses both. CPU clocks are not pinned
(`schedutil` governor); in the second timing 21% of schedules moved by more
than 5% while their ranking held, and the first repeats of a schedule run
slightly slow. Analytical features are estimates and omit hardware counters.
The study measures ranking quality, not the speed of a complete tuning
search.

## References

Pruner (ASPLOS 2025), Ansor (OSDI 2020), TVM (OSDI 2018), MetaSchedule
(NeurIPS 2022), TenSet (NeurIPS Datasets and Benchmarks 2021), TLP (ASPLOS
2023), XGBoost (KDD 2016).
