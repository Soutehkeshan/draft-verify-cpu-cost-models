
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
    defaults = AnalyticConstants(**target.analytic_constants())
    positions = [np.asarray(value, dtype=float) for value in train["vector"]]
    layout_values = [layouts[int(value)] for value in train["sketch_id"]]
    fitted = fit_constants(
        positions,
        layout_values,
        train["median_ms"].to_numpy(dtype=float),
        initial=defaults,
    )
    fitted_names = ("peak_flops", "bandwidth_bytes_s")
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
        "fitted_constants": list(fitted_names),
        "fit_multipliers": fit_multipliers,
        "hand_set_constants": defaults.to_dict(),
    }
    write_json(results_dir / "constants.json", payload, provenance)
    print(f"Fitted analytic constants on {len(train)} training schedules")
    for name in fitted_names:
        print(
            f"{name}: {getattr(defaults, name):.6g} -> "
            f"{getattr(fitted, name):.6g} ({fit_multipliers[name]:.3g}x)"
        )


if __name__ == "__main__":
    main()
