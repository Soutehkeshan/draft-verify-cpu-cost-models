
#!/usr/bin/env python3
"""Create deterministic stratified train/validation/test schedule splits."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.study import DATA_DIR, load_pool, save_json


def main() -> None:
    frame = load_pool()
    rng = np.random.default_rng(0)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for _, group in frame.groupby("sketch_id", sort=True):
        ids = group["id"].astype(int).to_numpy()
        rng.shuffle(ids)
        n_train = int(len(ids) * 0.70)
        n_validation = int(len(ids) * 0.15)
        train.extend(ids[:n_train].tolist())
        validation.extend(ids[n_train : n_train + n_validation].tolist())
        test.extend(ids[n_train + n_validation :].tolist())

    subset_order = np.asarray(train, dtype=int)
    rng.shuffle(subset_order)
    sizes = [100, 250, 500, 1000, 2000]
    nested = {
        str(size): subset_order[: min(size, len(subset_order))].tolist()
        for size in sizes
    }
    nested["all"] = subset_order.tolist()
    splits = {
        "seed": 0,
        "train": sorted(train),
        "validation": sorted(validation),
        "test": sorted(test),
        "nested_train": nested,
    }
    save_json(DATA_DIR / "splits.json", splits)
    print(
        f"Split {len(frame)} valid samples: "
        f"train={len(train)}, validation={len(validation)}, test={len(test)}"
    )


if __name__ == "__main__":
    main()
