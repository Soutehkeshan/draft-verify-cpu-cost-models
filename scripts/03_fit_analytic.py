
#!/usr/bin/env python3
"""Fit analytic hardware constants using only the training split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.analytic import AnalyticConstants, fit_constants
from dvcm.config import (
    add_target_argument,
    experiment_metadata,
    load_target_config,
    load_workload_config,
)
from dvcm.provenance import capture_provenance, write_json
from dvcm.study import (
    load_layouts,
    load_pool,
    load_splits,
    validate_splits,
)

FITTED_NAMES = ("peak_flops", "bandwidth_bytes_s")


def _fit_frame(frame, layouts, initial):
    return fit_constants(
        [np.asarray(value, dtype=float) for value in frame["vector"]],
        [layouts[int(value)] for value in frame["sketch_id"]],
        frame["median_ms"].to_numpy(dtype=float),
        initial=initial,
    )


def fit_nested_constants(train, nested_train, layouts, initial, train_fit):
    """Refit the fitted constants on each nested TRAIN subset, rows in pool order.

    A subset that equals TRAIN reuses ``train_fit``, the fit of identical
    inputs, so the full-TRAIN models keep exactly the same analytic feature.
    """
    train_ids = {int(value) for value in train["id"]}
    nested = {}
    for size_name, subset_ids in nested_train.items():
        ids = {int(value) for value in subset_ids}
        subset = train[train["id"].astype(int).isin(ids)]
        if subset.empty:
            continue
        fitted = train_fit if ids == train_ids else _fit_frame(subset, layouts, initial)
        nested[size_name] = {
            "training_samples": len(subset),
            "constants": fitted.to_dict(),
            "fit_multipliers": {
                name: getattr(fitted, name) / getattr(initial, name) for name in FITTED_NAMES
            },
        }
    return nested


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
    train_ids = set(int(value) for value in splits["train"])
    train = frame[frame["id"].astype(int).isin(train_ids)]
    layouts = load_layouts(data_dir, target)
    defaults = AnalyticConstants.from_target(target)
    fitted = _fit_frame(train, layouts, defaults)
    fitted_names = FITTED_NAMES
    fit_multipliers = {
        name: getattr(fitted, name) / getattr(defaults, name)
        for name in fitted_names
    }
    payload = {
        "target_id": target.target_id,
        "workload_id": workload.workload_id,
        "mode": "fitted",
        "seed": 0,
        "training_samples": len(train),
        "constants": fitted.to_dict(),
        "initial_fitted_constants": {
            name: getattr(defaults, name) for name in fitted_names
        },
        "initial_fitted_constants_note": "initial guess, fitted on TRAIN only",
        "fitted_constants": list(fitted_names),
        "fit_multipliers": fit_multipliers,
        "hand_set_constants": defaults.to_dict(),
        "nested_train_constants": fit_nested_constants(
            train, splits["nested_train"], layouts, defaults, fitted
        ),
        "nested_train_constants_note": (
            "peak_flops and bandwidth_bytes_s refit on each nested TRAIN subset only; "
            "the learning-curve models of a subset use its constants"
        ),
    }
    write_json(results_dir / "constants.json", payload, provenance)
    print(f"Fitted analytic constants on {len(train)} training schedules")
    for name in fitted_names:
        print(
            f"{name}: {getattr(defaults, name):.6g} -> "
            f"{getattr(fitted, name):.6g} ({fit_multipliers[name]:.3g}x)"
        )
    for size_name, entry in payload["nested_train_constants"].items():
        print(
            f"nested {size_name} ({entry['training_samples']} schedules): "
            + ", ".join(
                f"{name} {entry['fit_multipliers'][name]:.3g}x" for name in fitted_names
            )
        )


if __name__ == "__main__":
    main()
