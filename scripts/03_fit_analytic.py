
#!/usr/bin/env python3
"""Fit analytic hardware constants using only the training split."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.analytic import AnalyticConstants, fit_constants
from dvcm.study import (
    RESULTS_DIR,
    load_layouts,
    load_pool,
    load_splits,
    save_json,
    validate_splits,
)


def main() -> None:
    frame = load_pool()
    splits = load_splits()
    validate_splits(splits, set(frame["id"].astype(int)))
    train_ids = set(int(value) for value in splits["train"])
    train = frame[frame["id"].astype(int).isin(train_ids)]
    layouts = load_layouts()
    positions = [np.asarray(value, dtype=float) for value in train["vector"]]
    layout_values = [layouts[int(value)] for value in train["sketch_id"]]
    fitted = fit_constants(
        positions,
        layout_values,
        train["median_ms"].to_numpy(dtype=float),
    )
    defaults = AnalyticConstants()
    fitted_names = ("peak_flops", "bandwidth_bytes_s")
    fit_multipliers = {
        name: getattr(fitted, name) / getattr(defaults, name)
        for name in fitted_names
    }
    payload = {
        "mode": "fitted",
        "seed": 0,
        "training_samples": len(train),
        "constants": fitted.to_dict(),
        "initial_fitted_constants": {
            name: getattr(defaults, name) for name in fitted_names
        },
        "fitted_constants": list(fitted_names),
        "fit_multipliers": fit_multipliers,
        "hand_set_constants": {
            name: getattr(defaults, name)
            for name in (
                "l1_bytes",
                "l2_bytes",
                "lane_width",
                "num_cores",
                "vector_registers",
            )
        },
    }
    save_json(RESULTS_DIR / "constants.json", payload)
    print(f"Fitted analytic constants on {len(train)} training schedules")
    for name in fitted_names:
        print(
            f"{name}: {getattr(defaults, name):.6g} -> "
            f"{getattr(fitted, name):.6g} ({fit_multipliers[name]:.3g}x)"
        )


if __name__ == "__main__":
    main()
