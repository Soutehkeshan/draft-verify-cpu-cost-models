
#!/usr/bin/env python3
"""Fit analytic hardware constants using only the training split."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.analytic import AnalyticConstants, fit_constants
from dvcm.study import RESULTS_DIR, load_layouts, load_pool, load_splits, save_json


def main() -> None:
    frame = load_pool()
    splits = load_splits()
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
    payload = {
        "mode": "fitted",
        "seed": 0,
        "training_samples": len(train),
        "constants": fitted.to_dict(),
        "hand_set_constants": AnalyticConstants().to_dict(),
    }
    save_json(RESULTS_DIR / "constants.json", payload)
    print(f"Fitted analytic constants on {len(train)} training schedules")


if __name__ == "__main__":
    main()
