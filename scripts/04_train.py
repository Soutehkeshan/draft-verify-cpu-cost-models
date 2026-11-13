
#!/usr/bin/env python3
"""Tune XGBoost settings on validation data and train nested learning curves."""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.analytic import AnalyticConstants, latency_seconds
from dvcm.config import (
    add_target_argument,
    experiment_metadata,
    load_target_config,
    load_workload_config,
)
from dvcm.features import features
from dvcm.learned import train_xgb
from dvcm.provenance import capture_provenance, write_companion, write_json
from dvcm.study import (
    load_layouts,
    load_pool,
    load_splits,
    validate_splits,
)


def _feature_matrix(frame, layouts, include_derived, constants):
    matrices = []
    for row in frame.itertuples(index=False):
        position = np.asarray(row.vector, dtype=float)
        layout = layouts[int(row.sketch_id)]
        values = features(position, int(row.sketch_id), layout, include_derived)
        if include_derived:
            analytic_ms = latency_seconds(position, layout, constants) * 1000.0
            values = np.append(values, analytic_ms)
        matrices.append(values)
    return np.asarray(matrices, dtype=float)


def _rmse(y_true, prediction):
    return float(np.sqrt(np.mean((np.log(y_true) - prediction) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    add_target_argument(parser)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    target = load_target_config(args.target, ROOT)
    workload = load_workload_config(root=ROOT)
    data_dir, results_dir = target.paths(ROOT)
    provenance = capture_provenance(
        ROOT,
        allow_dirty=args.allow_dirty,
        run_context=experiment_metadata(target, workload, data_dir),
    )

    frame = load_pool(data_dir=data_dir)
    splits = load_splits(data_dir)
    validate_splits(splits, set(frame["id"].astype(int)))
    layouts = load_layouts(data_dir, target)
    constants_path = results_dir / "constants.json"
    if not constants_path.exists():
        raise FileNotFoundError("Run scripts/03_fit_analytic.py before training")
    import json

    with constants_path.open(encoding="utf-8") as source:
        constants = AnalyticConstants(**json.load(source)["constants"])

    train_ids = set(int(value) for value in splits["train"])
    validation_ids = set(int(value) for value in splits["validation"])
    train_frame = frame[frame["id"].astype(int).isin(train_ids)].reset_index(drop=True)
    validation_frame = frame[frame["id"].astype(int).isin(validation_ids)].reset_index(drop=True)
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Both train and validation splits need modelable measurements")

    grid = list(
        itertools.product(
            (3, 6),
            (0.03, 0.1),
            (100, 300),
        )
    )
    model_dir = results_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_configs = {}
    for feature_set, include_derived in (("base", False), ("base_analytic", True)):
        x_train = _feature_matrix(train_frame, layouts, include_derived, constants)
        x_val = _feature_matrix(validation_frame, layouts, include_derived, constants)
        y_train = train_frame["median_ms"].to_numpy(dtype=float)
        y_val = validation_frame["median_ms"].to_numpy(dtype=float)
        validation_results = []
        for depth, learning_rate, n_estimators in grid:
            model = train_xgb(
                x_train,
                y_train,
                max_depth=depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                seed=0,
            )
            prediction = model.predict(x_val)
            validation_results.append(
                {
                    "max_depth": depth,
                    "learning_rate": learning_rate,
                    "n_estimators": n_estimators,
                    "validation_log_rmse": _rmse(y_val, prediction),
                }
            )
        best = min(validation_results, key=lambda result: result["validation_log_rmse"])
        best_configs[feature_set] = best

        for size_name, subset_ids in splits["nested_train"].items():
            subset_id_set = set(int(value) for value in subset_ids)
            subset = train_frame[train_frame["id"].astype(int).isin(subset_id_set)]
            if subset.empty:
                continue
            x_subset = _feature_matrix(subset, layouts, include_derived, constants)
            for seed in range(5):
                model = train_xgb(
                    x_subset,
                    subset["median_ms"].to_numpy(dtype=float),
                    max_depth=best["max_depth"],
                    learning_rate=best["learning_rate"],
                    n_estimators=best["n_estimators"],
                    seed=seed,
                )
                model_path = model_dir / f"{feature_set}_n{size_name}_seed{seed}.json"
                model.get_booster().set_attr(
                    provenance=json.dumps(provenance, sort_keys=True)
                )
                model.save_model(model_path)
                write_companion(model_path, provenance)
    write_json(
        results_dir / "training_grid.json",
        {
            "grid": grid,
            "best": best_configs,
            "training_settings": {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "target_id": target.target_id,
                "workload_id": workload.workload_id,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_seeds": list(range(5)),
            },
        },
        provenance,
    )
    print(
        f"Trained 5-seed nested models using {len(train_frame)} training "
        f"and {len(validation_frame)} validation samples"
    )


if __name__ == "__main__":
    main()
