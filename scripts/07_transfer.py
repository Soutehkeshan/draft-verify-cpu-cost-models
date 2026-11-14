#!/usr/bin/env python3
"""Score M2-trained, Nano-only and M2-adapted XGBoost rankers on the Nano TEST set.

(a) zero-shot: the M2 models of 04_train.py (all M2 TRAIN rows, settings chosen
    on M2 VALIDATION) rank the Nano TEST schedules;
(b) reference: the Nano TEST schedules ranked by their measured M2 latency (this
    uses the M2 measurements of the TEST schedules; it is not a model);
(c) Nano-only: XGBoost on the first n Nano TRAIN rows (the nested subsets), with
    settings chosen per n and model on Nano VALIDATION;
(d) adapted: (c) plus the M2 model's predicted latency as one extra feature.

Only M2 rows of TRAIN ids are trained on. The M2 feature of a Nano TRAIN row is
an out-of-fold prediction (5 folds over M2 TRAIN, analytic constants refit
inside each fold), so a row's own M2 label never enters its feature; VALIDATION
and TEST rows use the full M2 models of (a). The Nano analytic constants of a
hybrid are those refit on the same n rows by 03_fit_analytic.py. Requires
`make eval` for both targets at the current commit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.analysis import (
    REPORTED_METRICS,
    paired_difference,
    resampled_metrics_parallel,
    seed_mean_interval,
)
from dvcm.analytic import AnalyticConstants, constants_from_payload, fit_constants
from dvcm.config import experiment_metadata, load_target_config, load_workload_config
from dvcm.learned import HYPERPARAMETER_GRID, train_xgb
from dvcm.metrics import ranking_metrics
from dvcm.provenance import capture_provenance, write_csv, write_json
from dvcm.study import load_layouts, load_pool, load_splits, validate_splits


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"dvcm_script_{Path(name).stem}", ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = _load_script("04_train.py")  # the training script's feature matrix and log-RMSE
EVALUATE = _load_script("05_evaluate.py")  # the evaluation script's TEST scopes
SOURCE_TARGET = "apple_m2"
NANO_TARGET = "jetson_nano"
NANO_SIZES = ("100", "250", "500", "1000")
SEEDS = tuple(range(5))
OOF_FOLDS = 5
OOF_SEED = 0
RESAMPLES = 2000
FEATURE_SETS = (("base", False, "xgb(base)"), ("base_analytic", True, "xgb(base+analytic)"))
ROLES = ("train", "validation", "test")


def m2_training_ids(m2_splits: dict[str, Any], nano_splits: dict[str, Any]) -> list[int]:
    """M2 TRAIN ids, after checking that both targets share every split role and subset."""
    for name in ROLES:
        if sorted(map(int, m2_splits[name])) != sorted(map(int, nano_splits[name])):
            raise ValueError(f"The M2 and Nano splits do not share the {name} role")
    m2_nested = {name: list(map(int, ids)) for name, ids in m2_splits["nested_train"].items()}
    nano_nested = {name: list(map(int, ids)) for name, ids in nano_splits["nested_train"].items()}
    if m2_nested != nano_nested:
        raise ValueError("The M2 and Nano splits do not share the nested TRAIN subsets")
    return sorted(int(value) for value in m2_splits["train"])


def require_disjoint(training_ids, splits: dict[str, Any]) -> None:
    """Refuse M2 training rows whose ids are VALIDATION or TEST ids."""
    held_out = {int(value) for name in ("validation", "test") for value in splits[name]}
    overlap = sorted({int(value) for value in training_ids} & held_out)
    if overlap:
        raise ValueError(f"M2 training rows include VALIDATION/TEST ids: {overlap[:5]}")


def oof_folds(ids, n_folds: int = OOF_FOLDS, seed: int = OOF_SEED) -> dict[int, int]:
    """Fold of every id: a seeded permutation of the sorted ids dealt round-robin."""
    order = np.random.default_rng(seed).permutation(np.asarray(sorted(int(v) for v in ids)))
    return {int(sample_id): position % n_folds for position, sample_id in enumerate(order)}


def out_of_fold_predictions(
    ids, folds: dict[int, int], fit_predict: Callable[[int, list[int], list[int]], dict[int, float]]
) -> dict[int, float]:
    """Predict each fold with a model fit on the other folds only."""
    predictions: dict[int, float] = {}
    for fold in sorted(set(folds.values())):
        held_out = [int(i) for i in ids if folds[int(i)] == fold]
        training = [int(i) for i in ids if folds[int(i)] != fold]
        predictions.update(fit_predict(fold, training, held_out))
    return predictions


def select_config(grid, score: Callable[[tuple], float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Lowest validation log-RMSE over the grid; the first configuration wins ties."""
    results = [
        {"max_depth": depth, "learning_rate": rate, "n_estimators": trees,
         "validation_log_rmse": float(score((depth, rate, trees)))}
        for depth, rate, trees in grid
    ]
    return dict(min(results, key=lambda result: result["validation_log_rmse"])), results


def _require_commit(paths: list[Path], commit: str) -> None:
    stale = {}
    for path in paths:
        recorded = (json.loads(path.read_text(encoding="utf-8")).get("provenance") or {}).get("git_commit")
        if recorded != commit:
            stale[str(path.relative_to(ROOT))] = recorded
    if stale:
        raise SystemExit(
            f"Refusing to run: inputs not produced at commit {commit[:12]} "
            f"(run make eval for apple_m2 and jetson_nano first): {stale}"
        )


def _rows(pool, ids):
    """Rows of the given ids in pool order, as the training script selects them."""
    return pool[pool["id"].astype(int).isin({int(value) for value in ids})].reset_index(drop=True)


def _fit_xgb(x, y, config, seed):
    return train_xgb(
        x, y,
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(config["n_estimators"]),
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    workload = load_workload_config(root=ROOT)
    source = load_target_config(SOURCE_TARGET, ROOT)
    target = load_target_config(NANO_TARGET, ROOT)
    source_data, source_results = source.paths(ROOT)
    target_data, target_results = target.paths(ROOT)
    provenance = capture_provenance(
        ROOT,
        allow_dirty=args.allow_dirty,
        run_context={
            "source": experiment_metadata(source, workload, source_data),
            "target": experiment_metadata(target, workload, target_data),
        },
    )
    commit = provenance["git_commit"]

    source_pool = load_pool(data_dir=source_data)
    target_pool = load_pool(data_dir=target_data)
    source_splits = load_splits(source_data)
    target_splits = load_splits(target_data)
    validate_splits(source_splits, set(source_pool["id"].astype(int)))
    validate_splits(target_splits, set(target_pool["id"].astype(int)))
    training_ids = m2_training_ids(source_splits, target_splits)
    require_disjoint(training_ids, target_splits)
    source_by_id = source_pool.set_index("id")
    for row in target_pool.itertuples(index=False):
        twin = source_by_id.loc[int(row.id)]
        if int(twin["sketch_id"]) != int(row.sketch_id) or list(map(int, twin["vector"])) != list(map(int, row.vector)):
            raise ValueError(f"Schedule {row.id} differs between the M2 and Nano pools")
    source_layouts = load_layouts(source_data, source)
    target_layouts = load_layouts(target_data, target)

    model_paths = {
        (feature_set, seed): source_results / "models" / f"{feature_set}_nall_seed{seed}.json"
        for feature_set, _, _ in FEATURE_SETS
        for seed in SEEDS
    }
    _require_commit(
        [
            source_results / "constants.json",
            source_results / "training_grid.json",
            target_results / "constants.json",
            target_results / "training_grid.json",
            target_results / "learning_curve.meta.json",
            *[path.with_suffix(".meta.json") for path in model_paths.values()],
        ],
        commit,
    )
    _, source_fitted, _ = constants_from_payload(
        json.loads((source_results / "constants.json").read_text(encoding="utf-8"))
    )
    source_best = json.loads((source_results / "training_grid.json").read_text(encoding="utf-8"))["best"]
    _, _, target_nested = constants_from_payload(
        json.loads((target_results / "constants.json").read_text(encoding="utf-8"))
    )
    target_best = json.loads((target_results / "training_grid.json").read_text(encoding="utf-8"))["best"]

    test = _rows(target_pool, target_splits["test"])
    validation = _rows(target_pool, target_splits["validation"])
    train = _rows(target_pool, target_splits["train"])
    source_train = _rows(source_pool, training_ids)
    y_test = test["median_ms"].to_numpy(dtype=float)
    y_validation = validation["median_ms"].to_numpy(dtype=float)

    def source_rows(ids):  # M2 rows in the given id order
        return source_by_id.loc[[int(value) for value in ids]].reset_index()

    features_identical = bool(
        np.array_equal(
            TRAIN._feature_matrix(test, target_layouts, True, source_fitted)[:, :-1],
            TRAIN._feature_matrix(source_rows(test["id"]), source_layouts, True, source_fitted)[:, :-1],
        )
    )

    # (a) the M2 models' predictions for the Nano VALIDATION and TEST schedules.
    zero_shot = {}
    for feature_set, include_derived, _ in FEATURE_SETS:
        x_validation = TRAIN._feature_matrix(
            source_rows(validation["id"]), source_layouts, include_derived, source_fitted
        )
        x_test = TRAIN._feature_matrix(source_rows(test["id"]), source_layouts, include_derived, source_fitted)
        for seed in SEEDS:
            model = XGBRegressor()
            model.load_model(model_paths[(feature_set, seed)])
            zero_shot[(feature_set, seed)] = {
                "validation": np.exp(model.predict(x_validation)),
                "test": np.exp(model.predict(x_test)),
            }

    # Out-of-fold M2 predictions for the TRAIN ids (the Nano TRAIN rows of (d)).
    folds = oof_folds(training_ids)
    source_initial = AnalyticConstants.from_target(source)
    fold_ids = source_train["id"].astype(int).map(folds)
    fold_constants = {}
    for fold in range(OOF_FOLDS):
        fold_rows = source_train[fold_ids != fold]
        fold_constants[fold] = fit_constants(
            [np.asarray(value, dtype=float) for value in fold_rows["vector"]],
            [source_layouts[int(value)] for value in fold_rows["sketch_id"]],
            fold_rows["median_ms"].to_numpy(dtype=float),
            initial=source_initial,
        )
    out_of_fold = {}
    for feature_set, include_derived, _ in FEATURE_SETS:
        for seed in SEEDS:
            def fit_predict(fold, fit_ids, predict_ids, include_derived=include_derived,
                            config=source_best[feature_set], seed=seed):
                constants = fold_constants[fold] if include_derived else source_fitted
                fit_rows = _rows(source_train, fit_ids)
                predict_rows = _rows(source_train, predict_ids)
                model = _fit_xgb(
                    TRAIN._feature_matrix(fit_rows, source_layouts, include_derived, constants),
                    fit_rows["median_ms"].to_numpy(dtype=float),
                    config,
                    seed,
                )
                predicted = np.exp(
                    model.predict(
                        TRAIN._feature_matrix(predict_rows, source_layouts, include_derived, constants)
                    )
                )
                return dict(zip(predict_rows["id"].astype(int), predicted))

            out_of_fold[(feature_set, seed)] = out_of_fold_predictions(training_ids, folds, fit_predict)

    # (c) Nano-only and (d) adapted, settings chosen per n on Nano VALIDATION.
    predictions: dict[tuple, np.ndarray] = {}
    settings: dict[tuple, dict[str, Any]] = {}
    selection = []
    for size in NANO_SIZES:
        subset = _rows(train, target_splits["nested_train"][size])
        y_subset = subset["median_ms"].to_numpy(dtype=float)
        for feature_set, include_derived, _ in FEATURE_SETS:
            constants = target_nested[size]
            x_subset = TRAIN._feature_matrix(subset, target_layouts, include_derived, constants)
            x_validation = TRAIN._feature_matrix(validation, target_layouts, include_derived, constants)
            x_test = TRAIN._feature_matrix(test, target_layouts, include_derived, constants)
            for variant in ("c_nano_only", "d_adapted"):
                def inputs(seed, variant=variant, feature_set=feature_set, x_subset=x_subset,
                           x_validation=x_validation, x_test=x_test, subset=subset):
                    if variant == "c_nano_only":
                        return x_subset, x_validation, x_test
                    m2_feature = np.asarray(
                        [out_of_fold[(feature_set, seed)][int(i)] for i in subset["id"]]
                    )
                    return (
                        np.column_stack([x_subset, m2_feature]),
                        np.column_stack([x_validation, zero_shot[(feature_set, seed)]["validation"]]),
                        np.column_stack([x_test, zero_shot[(feature_set, seed)]["test"]]),
                    )

                fit_x, validation_x, _ = inputs(SEEDS[0])
                best, results = select_config(
                    HYPERPARAMETER_GRID,
                    lambda config: TRAIN._rmse(
                        y_validation,
                        _fit_xgb(
                            fit_x, y_subset,
                            dict(zip(("max_depth", "learning_rate", "n_estimators"), config)),
                            SEEDS[0],
                        ).predict(validation_x),
                    ),
                )
                selection.append(
                    {"variant": variant, "feature_set": feature_set, "nano_train_rows": len(subset),
                     "best": best, "grid_results": results}
                )
                settings[(variant, feature_set, size)] = best
                for seed in SEEDS:
                    fit_x, _, test_x = inputs(seed)
                    model = _fit_xgb(fit_x, y_subset, best, seed)
                    predictions[(variant, feature_set, size, seed)] = np.exp(model.predict(test_x))
    for feature_set, _, _ in FEATURE_SETS:
        settings[("a_zero_shot", feature_set, "m2")] = source_best[feature_set]
        for seed in SEEDS:
            predictions[("a_zero_shot", feature_set, "m2", seed)] = zero_shot[(feature_set, seed)]["test"]
    predictions[("b_m2_measured", "none", "none", 0)] = source_rows(test["id"])["median_ms"].to_numpy(dtype=float)

    def describe(variant, feature_set, size):
        label = dict((name, text) for name, _, text in FEATURE_SETS).get(feature_set)
        config = settings.get((variant, feature_set, size), {})
        info = {
            "a_zero_shot": (f"{label} trained on M2 (zero-shot)", len(training_ids), 0, "M2 TRAIN"),
            "b_m2_measured": ("measured M2 latency (reference)", 0, 0,
                              "M2 measurements of the TEST schedules (reference only)"),
            "c_nano_only": (f"{label} trained on Nano only", 0, int(size) if size.isdigit() else 0, "none"),
            "d_adapted": (f"{label} + M2-prediction feature", len(training_ids),
                          int(size) if size.isdigit() else 0,
                          "M2 TRAIN (out-of-fold for the Nano TRAIN rows)"),
        }[variant]
        return {
            "variant": variant,
            "model": info[0],
            "feature_set": feature_set,
            "m2_train_rows": info[1],
            "nano_train_rows": info[2],
            "m2_labels_used": info[3],
            "max_depth": config.get("max_depth"),
            "learning_rate": config.get("learning_rate"),
            "n_estimators": config.get("n_estimators"),
            "settings_chosen_on": {
                "a_zero_shot": "M2 VALIDATION", "b_m2_measured": "n/a",
                "c_nano_only": "Nano VALIDATION", "d_adapted": "Nano VALIDATION",
            }[variant],
        }

    scopes = list(EVALUATE._scope_indices(test))
    keys = list(predictions)
    groups: dict[tuple, list[tuple]] = {}
    for key in keys:
        groups.setdefault(key[:3], []).append(key)
    tasks = [
        (y_test[indexes], predictions[key][indexes], 20260 + scope_index)
        for scope_index, (_, indexes) in enumerate(scopes)
        for key in keys
    ]
    resampled_values = iter(resampled_metrics_parallel(tasks, RESAMPLES))
    resampled = {(scope, key): next(resampled_values) for scope, _ in scopes for key in keys}

    seed_rows, summary_rows, paired_rows = [], [], []
    for scope, indexes in scopes:
        truth = y_test[indexes]
        point = {key: ranking_metrics(truth, predictions[key][indexes]) for key in keys}
        for group_key, members in groups.items():
            info = describe(*group_key)
            for key in members:
                seed_rows.append(
                    {"scope": scope, **info, "seed": key[3],
                     **{name: point[key][name] for name in REPORTED_METRICS}}
                )
            summary = seed_mean_interval(
                [point[key] for key in members], [resampled[(scope, key)] for key in members]
            )
            summary_rows.append(
                {
                    "scope": scope, "n_test": len(indexes), **info, "seed_count": len(members),
                    **{f"{name}_{statistic}": value
                       for name, statistics in summary.items()
                       for statistic, value in statistics.items()},
                    "bootstrap_resamples": RESAMPLES,
                }
            )
        for feature_set, _, label in FEATURE_SETS:
            for size in NANO_SIZES:
                nano_only = groups[("c_nano_only", feature_set, size)]
                adapted = groups[("d_adapted", feature_set, size)]
                differences = paired_difference(
                    [point[key] for key in nano_only],
                    [point[key] for key in adapted],
                    [resampled[(scope, key)] for key in nano_only],
                    [resampled[(scope, key)] for key in adapted],
                    labels=("nano_only", "adapted"),
                )
                for name in REPORTED_METRICS:
                    paired_rows.append(
                        {
                            "scope": scope, "n_test": len(indexes), "model": label,
                            "feature_set": feature_set, "nano_train_rows": int(size),
                            "metric": name,
                            "difference_definition": "(d) M2-adapted minus (c) Nano-only, seed paired",
                            **differences[name],
                            "bootstrap_resamples": RESAMPLES, "training_seeds": len(SEEDS),
                        }
                    )

    # When (c) picked the full-TRAIN setting, its models are the learning-curve models.
    learning = EVALUATE.pd.read_csv(target_results / "learning_curve.csv")
    learning = learning[learning["scope"] == "pooled"].set_index(["feature_set", "subset", "seed"])
    consistency = []
    for feature_set, _, _ in FEATURE_SETS:
        for size in NANO_SIZES:
            chosen = settings[("c_nano_only", feature_set, size)]
            same_setting = all(chosen[name] == target_best[feature_set][name]
                               for name in ("max_depth", "learning_rate", "n_estimators"))
            matches = None
            if same_setting:
                matches = all(
                    ranking_metrics(y_test, predictions[("c_nano_only", feature_set, size, seed)])["spearman"]
                    == learning.loc[(feature_set, size, seed), "spearman"]
                    for seed in SEEDS
                )
            consistency.append({"feature_set": feature_set, "nano_train_rows": int(size),
                                "same_setting_as_full_train_grid": same_setting,
                                "spearman_equals_learning_curve": matches})

    write_csv(EVALUATE.pd.DataFrame(seed_rows), target_results / "transfer_seeds.csv", provenance)
    write_csv(EVALUATE.pd.DataFrame(summary_rows), target_results / "transfer_summary.csv", provenance)
    write_csv(EVALUATE.pd.DataFrame(paired_rows), target_results / "transfer_paired.csv", provenance)
    write_json(
        target_results / "transfer_grid.json",
        {
            "description": __doc__,
            "grid": [list(config) for config in HYPERPARAMETER_GRID],
            "selection_criterion": "Nano VALIDATION log-RMSE of a seed-0 model; the first setting wins ties",
            "selection": selection,
            "m2_models": {
                feature_set: {"settings": source_best[feature_set],
                              "settings_chosen_on": "M2 VALIDATION (04_train.py)",
                              "training_rows": len(training_ids)}
                for feature_set, _, _ in FEATURE_SETS
            },
            "out_of_fold": {
                "folds": OOF_FOLDS,
                "seed": OOF_SEED,
                "rule": "seeded permutation of the sorted M2 TRAIN ids dealt round-robin",
                "fold_sizes": [int((fold_ids == fold).sum()) for fold in range(OOF_FOLDS)],
                "fold_constants": {
                    str(fold): {"peak_flops": value.peak_flops, "bandwidth_bytes_s": value.bandwidth_bytes_s}
                    for fold, value in fold_constants.items()
                },
            },
            "leakage_checks": {
                "m2_training_rows": len(training_ids),
                "m2_training_ids_in_validation": len(set(training_ids) & set(map(int, target_splits["validation"]))),
                "m2_training_ids_in_test": len(set(training_ids) & set(map(int, target_splits["test"]))),
                "schedules_identical_across_targets": True,
                "schedule_features_identical_across_targets": features_identical,
            },
            "learning_curve_consistency": consistency,
        },
        provenance,
    )

    pooled = [row for row in summary_rows if row["scope"] == "pooled"]
    for row in pooled:
        print(
            f"{row['variant']:14s} {row['model']:45s} n={row['nano_train_rows']:>4} "
            f"Spearman {row['spearman_mean']:.3f} [{row['spearman_bootstrap95_low']:.3f}, "
            f"{row['spearman_bootstrap95_high']:.3f}]  top-5 {row['top_5_latency_ratio_mean']:.3f}"
        )
    for row in paired_rows:
        if row["scope"] == "pooled" and row["metric"] in ("spearman", "top_5_latency_ratio"):
            print(
                f"(d)-(c) {row['model']:20s} n={row['nano_train_rows']:>4} {row['metric']:22s} "
                f"{row['difference_mean_over_seeds']:+.4f} [{row['paired_bootstrap_ci_low']:+.4f}, "
                f"{row['paired_bootstrap_ci_high']:+.4f}]"
            )
    print(f"Learning-curve consistency: {consistency}")
    print(f"Leakage checks: M2 training ids in VALIDATION/TEST = 0; schedule features identical: {features_identical}")


if __name__ == "__main__":
    main()
