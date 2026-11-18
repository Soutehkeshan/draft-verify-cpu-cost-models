#!/usr/bin/env python3
"""Score the frozen models on the independent confirmation TEST set.

Written and committed before the confirmation schedules were timed. Nothing is
retrained or re-tuned: every model, setting and analytic constant comes from
results/jetson_nano/ as fitted on the original TRAIN set. The new schedules
(data/jetson_nano_confirm/, none of them in the original pool) are scored with
the original evaluation code plus one measure that grows with the test set,
top-1% recall in the model's top 3%.

Primary questions, fixed in advance (the gain from adding the analytical
features to XGBoost, both trained on all 3,499 TRAIN schedules, seed-paired,
2,000 paired resamples; a gain counts as confirmed when its 95% range excludes
zero in the expected direction):
  1. overall Spearman;
  2. fastest-decile Spearman;
  3. top-1% recall in the top 3%;
  4. top-1 latency ratio (lower is better).
Secondary: the gain at every nested TRAIN size, the draft test with the
fastest 1% as targets, and the repeat-noise ceiling of the new timings.

Writes to results/jetson_nano_confirm/: scores.csv, paired.csv,
learning_curve.csv, draft.csv, noise.csv and summary.json, each with provenance.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.analysis import bootstrap_samples, paired_difference, seed_mean_interval
from dvcm.analytic import latency_seconds
from dvcm.config import add_target_argument, experiment_metadata, load_target_config, load_workload_config
from dvcm.metrics import ranking_metrics, top_k_recall
from dvcm.provenance import capture_provenance, write_csv, write_json
from dvcm.study import load_layouts, load_pool

METRICS = (
    "spearman",
    "fastest_decile_spearman",
    "top_1pct_recall_in_3pct",
    "top_1_latency_ratio",
    "top_5_latency_ratio",
)
PRIMARY = ("spearman", "fastest_decile_spearman", "top_1pct_recall_in_3pct", "top_1_latency_ratio")
LOWER_IS_BETTER = {"top_1_latency_ratio", "top_5_latency_ratio"}
RESAMPLES = 2000
SEEDS = tuple(range(5))
FEATURE_SETS = (("base", "xgb(base)"), ("base_analytic", "xgb(base+analytic)"))
DRAFT_FRACTIONS = (0.01, 0.02, 0.04, 0.08, 0.16)


def confirmation_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """The original ranking metrics plus top-1% recall within the model's top 3%."""
    values = ranking_metrics(y_true, predicted)
    n = len(y_true)
    values["top_1pct_recall_in_3pct"] = top_k_recall(
        y_true, predicted, max(1, math.ceil(0.01 * n)), max(1, math.ceil(0.03 * n))
    )
    return {name: float(values[name]) for name in METRICS}


def _resampled(task: tuple) -> np.ndarray:
    y_true, predicted, seed, n_resamples = task
    rows = []
    for sample in bootstrap_samples(len(y_true), n_resamples, seed):
        scores = confirmation_metrics(y_true[sample], predicted[sample])
        rows.append([scores[name] for name in METRICS])
    return np.asarray(rows, dtype=float)


def resample_all(tasks: list[tuple], workers: int) -> list[np.ndarray]:
    if workers <= 1:
        return [_resampled(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_resampled, tasks))


def _evaluation_module():
    spec = importlib.util.spec_from_file_location("dvcm_evaluate", ROOT / "scripts" / "04_evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def confirmed(row: dict, metric: str) -> bool:
    if metric in LOWER_IS_BETTER:
        return row["paired_bootstrap_ci_high"] < 0
    return row["paired_bootstrap_ci_low"] > 0


def draft_rows(scope: str, y_true: np.ndarray, drafts: dict[str, np.ndarray]) -> list[dict]:
    """How many of the fastest 1% each drafter keeps in its best K, K a share of the set."""
    n = len(y_true)
    targets = set(np.argsort(y_true, kind="stable")[: max(1, math.ceil(0.01 * n))].tolist())
    rows = []
    for fraction in DRAFT_FRACTIONS:
        k = max(1, math.ceil(fraction * n))
        for name, scores in drafts.items():
            kept = len(targets & set(np.argsort(scores, kind="stable")[:k].tolist()))
            rows.append({"scope": scope, "n_test": n, "draft_fraction": fraction, "K": k, "drafter": name,
                         "targets": len(targets), "targets_kept": kept, "share_kept": kept / len(targets)})
        rows.append({"scope": scope, "n_test": n, "draft_fraction": fraction, "K": k, "drafter": "random (expected)",
                     "targets": len(targets), "targets_kept": len(targets) * k / n, "share_kept": k / n})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_target_argument(parser)
    parser.add_argument("--confirm-dir", type=Path, default=ROOT / "data" / "jetson_nano_confirm")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "jetson_nano_confirm")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    target = load_target_config(args.target, ROOT)
    workload = load_workload_config(root=ROOT)
    data_dir, results_dir = target.paths(ROOT)
    provenance = capture_provenance(
        ROOT, allow_dirty=args.allow_dirty, run_context=experiment_metadata(target, workload, args.confirm_dir)
    )
    evaluate = _evaluation_module()

    original_ids = set(load_pool(include_failures=True, data_dir=data_dir)["id"].astype(int))
    pool = load_pool(data_dir=args.confirm_dir).sort_values("id").reset_index(drop=True)
    all_rows = load_pool(include_failures=True, data_dir=args.confirm_dir)
    if original_ids & set(pool["id"].astype(int)):
        raise SystemExit("Confirmation schedules overlap the original pool")
    layouts = load_layouts(data_dir, target)
    hand_set, fitted, nested = evaluate._read_constants(results_dir)
    y_true = pool["median_ms"].to_numpy(dtype=float)
    scopes = list(evaluate._scope_indices(pool))
    args.output.mkdir(parents=True, exist_ok=True)

    sizes = [name for name in nested]
    predictions = {
        (feature_set, size, seed): evaluate._predict_model_seed(
            pool, layouts, nested[size] if feature_set == "base_analytic" else fitted,
            feature_set, size, seed, results_dir,
        )
        for feature_set, _ in FEATURE_SETS for size in sizes for seed in SEEDS
    }
    analytic = {
        name: np.asarray([latency_seconds(np.asarray(row.vector), layouts[int(row.sketch_id)], constants) * 1000
                          for row in pool.itertuples(index=False)])
        for name, constants in (("analytic(hand_set)", hand_set), ("analytic(fitted)", fitted))
    }

    # Resample tasks: every full-TRAIN model and both analytic models in every scope; every
    # nested-size model in the pooled scope. Seed 30260 + scope index, shared by all models of a scope.
    tasks, keys = [], []
    for scope_index, (scope, indexes) in enumerate(scopes):
        seed = 30260 + scope_index
        for key, values in predictions.items():
            if key[1] == "all" or scope == "pooled":
                tasks.append((y_true[indexes], values[indexes], seed, args.resamples))
                keys.append((scope, key))
        for name, values in analytic.items():
            tasks.append((y_true[indexes], values[indexes], seed, args.resamples))
            keys.append((scope, name))
    resampled = dict(zip(keys, resample_all(tasks, args.workers)))

    score_rows, paired_rows, curve_rows = [], [], []
    for scope_index, (scope, indexes) in enumerate(scopes):
        truth = y_true[indexes]
        rng = np.random.default_rng(40260 + scope_index)
        random_scores = [confirmation_metrics(truth, rng.random(len(truth))) for _ in range(1000)]
        score_rows.append({"scope": scope, "scorer": "random", "n_test": len(truth), **{
            f"{m}{suffix}": value for m in METRICS for suffix, value in (
                ("", float(np.mean([s[m] for s in random_scores]))),
                ("_low", float(np.quantile([s[m] for s in random_scores], 0.025))),
                ("_high", float(np.quantile([s[m] for s in random_scores], 0.975))))}})
        for name, values in analytic.items():
            point = confirmation_metrics(truth, values[indexes])
            low, high = np.quantile(resampled[(scope, name)], [0.025, 0.975], axis=0)
            score_rows.append({"scope": scope, "scorer": name, "n_test": len(truth), **{
                f"{m}{suffix}": value for i, m in enumerate(METRICS)
                for suffix, value in (("", point[m]), ("_low", low[i]), ("_high", high[i]))}})
        point = {key: confirmation_metrics(truth, values[indexes]) for key, values in predictions.items()
                 if key[1] == "all" or scope == "pooled"}
        for feature_set, model in FEATURE_SETS:
            model_keys = [(feature_set, "all", seed) for seed in SEEDS]
            summary = seed_mean_interval([point[k] for k in model_keys], [resampled[(scope, k)] for k in model_keys],
                                         metrics=METRICS)
            score_rows.append({"scope": scope, "scorer": model, "n_test": len(truth), **{
                f"{m}{suffix}": summary[m][stat] for m in METRICS
                for suffix, stat in (("", "mean"), ("_low", "bootstrap95_low"), ("_high", "bootstrap95_high"))}})
        for size in (sizes if scope == "pooled" else ["all"]):
            base_keys = [("base", size, seed) for seed in SEEDS]
            hybrid_keys = [("base_analytic", size, seed) for seed in SEEDS]
            differences = paired_difference(
                [point[k] for k in base_keys], [point[k] for k in hybrid_keys],
                [resampled[(scope, k)] for k in base_keys], [resampled[(scope, k)] for k in hybrid_keys],
                metrics=METRICS,
            )
            for metric in METRICS:
                row = {"scope": scope, "subset": size, "train_size": None,  # filled in from splits.json below
                       "n_test": len(truth), "metric": metric, "primary": metric in PRIMARY,
                       **differences[metric], "bootstrap_resamples": args.resamples}
                row["confirmed"] = confirmed(row, metric)
                (paired_rows if size == "all" else curve_rows).append(row)
                if size == "all" and scope == "pooled":
                    curve_rows.append(dict(row))

    splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
    for row in paired_rows + curve_rows:
        row["train_size"] = len(splits["nested_train"][row["subset"]])

    drafts = []
    ensemble_base = np.mean([predictions[("base", "all", seed)] for seed in SEEDS], axis=0)
    for scope, indexes in scopes:
        drafts += draft_rows(scope, y_true[indexes], {"xgb(base)": ensemble_base[indexes],
                                                       "analytic(fitted)": analytic["analytic(fitted)"][indexes]})
    noise = [evaluate._noise_row(scope, pool.iloc[indexes], "median_ms", "repeats_ms") for scope, indexes in scopes]

    write_csv(pd.DataFrame(score_rows), args.output / "scores.csv", provenance)
    write_csv(pd.DataFrame(paired_rows), args.output / "paired.csv", provenance)
    write_csv(pd.DataFrame(curve_rows).sort_values(["metric", "train_size"]), args.output / "learning_curve.csv", provenance)
    write_csv(pd.DataFrame(drafts), args.output / "draft.csv", provenance)
    write_csv(pd.DataFrame(noise), args.output / "noise.csv", provenance)
    pooled = {row["metric"]: row for row in paired_rows if row["scope"] == "pooled"}
    write_json(args.output / "summary.json", {
        "n_schedules_timed": int(len(all_rows)),
        "n_scored": int(len(pool)),
        "status_counts": {str(k): int(v) for k, v in all_rows["status"].value_counts().items()},
        "primary": {m: {"difference": pooled[m]["difference_mean_over_seeds"],
                        "ci95": [pooled[m]["paired_bootstrap_ci_low"], pooled[m]["paired_bootstrap_ci_high"]],
                        "confirmed": pooled[m]["confirmed"]} for m in PRIMARY},
        "models_from": str(results_dir.relative_to(ROOT)),
    }, provenance)
    for metric in PRIMARY:
        row = pooled[metric]
        print(f"{metric:28s} gain {row['difference_mean_over_seeds']:+.4f} "
              f"[{row['paired_bootstrap_ci_low']:+.4f}, {row['paired_bootstrap_ci_high']:+.4f}] "
              f"confirmed={row['confirmed']}")


if __name__ == "__main__":
    main()
