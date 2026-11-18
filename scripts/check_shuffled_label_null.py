#!/usr/bin/env python3
"""Leakage check: train XGBoost on shuffled TRAIN labels and score TEST.

For seeds 0..N-1 (default 120) the TRAIN latencies are permuted globally with
that seed, both feature sets are trained with the VALIDATION-chosen settings
and the TRAIN-fitted constants of 04_evaluate.py, and the pooled TEST
Spearman and fastest-decile Spearman are recorded. With no leak they average
about zero. It writes shuffled_label_null_extended.csv (one row per
permutation) and shuffled_label_null_extended_summary.csv (mean, standard
error, and the mean in standard errors, for seeds 0-19 and for all seeds;
seeds 0-19 are the 20 permutations 04_evaluate.py used to run itself).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.config import (
    add_target_argument,
    experiment_metadata,
    load_target_config,
    load_workload_config,
)
from dvcm.learned import train_xgb
from dvcm.metrics import ranking_metrics
from dvcm.provenance import capture_provenance, write_csv
from dvcm.study import load_layouts, load_pool, load_splits, validate_splits

MODELS = (("base", False, "xgb(base)"), ("base_analytic", True, "xgb(base+analytic)"))
METRICS = ("spearman", "fastest_decile_spearman")
EVALUATION_SEEDS = 20  # the permutations 04_evaluate.py used to run


def _evaluation_module():
    spec = importlib.util.spec_from_file_location("dvcm_evaluate", ROOT / "scripts" / "04_evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def permuted_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    """The global permutation of the TRAIN labels for this seed."""
    return np.random.default_rng(seed).permutation(np.asarray(labels))


def summarize(values) -> dict:
    values = np.asarray(values, dtype=float)
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    mean = float(np.mean(values))
    return {
        "n_permutations": int(values.size),
        "mean": mean,
        "standard_error": standard_error,
        "mean_over_standard_error": mean / standard_error,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_target_argument(parser)
    parser.add_argument("--permutations", type=int, default=120)
    parser.add_argument("--data-dir", type=Path, help="dataset directory (default: the profile's)")
    parser.add_argument("--results-dir", type=Path, help="evaluation outputs (default: the profile's)")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.permutations < EVALUATION_SEEDS:
        parser.error(f"--permutations must be at least {EVALUATION_SEEDS}")
    target = load_target_config(args.target, ROOT)
    workload = load_workload_config(root=ROOT)
    default_data, default_results = target.paths(ROOT)
    data_dir = (args.data_dir or default_data).resolve()
    results_dir = (args.results_dir or default_results).resolve()
    provenance = capture_provenance(
        ROOT, allow_dirty=args.allow_dirty, run_context=experiment_metadata(target, workload, data_dir)
    )
    evaluate = _evaluation_module()

    pool = load_pool(data_dir=data_dir)
    splits = load_splits(data_dir)
    validate_splits(splits, set(pool["id"].astype(int)))
    layouts = load_layouts(data_dir, target)
    train = pool[pool["id"].astype(int).isin(set(splits["train"]))].reset_index(drop=True)
    test = pool[pool["id"].astype(int).isin(set(splits["test"]))].reset_index(drop=True)
    _, fitted, _ = evaluate._read_constants(results_dir)
    grid = json.loads((results_dir / "training_grid.json").read_text(encoding="utf-8"))
    inputs_commit = {
        name: json.loads((results_dir / name).read_text(encoding="utf-8"))["provenance"]["git_commit"]
        for name in ("constants.json", "training_grid.json")
    }
    labels = train["median_ms"].to_numpy(dtype=float)
    y_true = test["median_ms"].to_numpy(dtype=float)

    rows = []
    for feature_set, include_derived, model_name in MODELS:
        x_train = evaluate._model_features(train, layouts, include_derived, fitted)
        x_test = evaluate._model_features(test, layouts, include_derived, fitted)
        config = grid["best"][feature_set]
        for seed in range(args.permutations):
            model = train_xgb(
                x_train,
                permuted_labels(labels, seed),
                max_depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                n_estimators=config["n_estimators"],
                seed=seed,
            )
            scores = ranking_metrics(y_true, np.exp(model.predict(x_test)))
            rows.append(
                {
                    "shuffle_mode": "global",
                    "model": model_name,
                    "scope": "pooled",
                    "seed": seed,
                    **{name: scores[name] for name in METRICS},
                    "max_depth": config["max_depth"],
                    "learning_rate": config["learning_rate"],
                    "n_estimators": config["n_estimators"],
                    "n_train": len(train),
                    "n_test": len(test),
                    "inputs_commit": inputs_commit["constants.json"],
                }
            )
    per_seed = pd.DataFrame(rows)
    write_csv(per_seed, results_dir / "shuffled_label_null_extended.csv", provenance)

    summary_rows = []
    for _, _, model_name in MODELS:
        group = per_seed[per_seed["model"] == model_name]
        for metric in METRICS:
            for label, subset in (
                (f"seeds 0-{EVALUATION_SEEDS - 1}", group[group["seed"] < EVALUATION_SEEDS]),
                (f"seeds 0-{args.permutations - 1}", group),
            ):
                summary_rows.append(
                    {"model": model_name, "metric": metric, "permutations": label, **summarize(subset[metric])}
                )
    summary = pd.DataFrame(summary_rows)
    write_csv(summary, results_dir / "shuffled_label_null_extended_summary.csv", provenance)
    for row in summary_rows:
        print(
            f"{args.target} {row['model']:18s} {row['metric']:23s} {row['permutations']:11s} "
            f"mean {row['mean']:+.4f}  SE {row['standard_error']:.4f}  "
            f"mean/SE {row['mean_over_standard_error']:+.2f}"
        )


if __name__ == "__main__":
    main()
