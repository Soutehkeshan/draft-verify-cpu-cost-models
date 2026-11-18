#!/usr/bin/env python3
"""Draw a fresh, independent set of schedules for a confirmation TEST set.

Samples schedules uniformly over the valid encoded space, exactly as the
original 5,000 were drawn (each sketch weighted by its number of valid
schedules, duplicates rejected), but with a different seed and excluding every
schedule already in data/jetson_nano/pool.parquet. New ids continue after the
largest existing id. Every new schedule is TEST: the models trained on the
original TRAIN set are scored on them unchanged.

Writes <output>/pool.parquet (id, sketch_id, vector), <output>/splits.json
(all ids in TEST, in a seeded random measurement order) and copies of the
original layouts, which 01_collect.py --positions-from reads.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.encoding import _slice, random_position, space_size
from dvcm.study import save_json

SOURCE = ROOT / "data" / "jetson_nano"


def load_layouts(directory: Path) -> list[dict]:
    """The original layouts, with JSON slices turned into slice objects."""
    layouts = []
    for sketch_id in range(len(list(directory.glob("layout_*.json")))):
        layout = json.loads((directory / f"layout_{sketch_id}.json").read_text(encoding="utf-8"))
        for tile in layout["tiles"]:
            tile["slice"] = _slice(tile["slice"])
        layouts.append(layout)
    return layouts


def sample_new(layouts: list[dict], exclude: set, count: int, seed: int) -> list[tuple[int, np.ndarray]]:
    """`count` unique schedules not in `exclude`, drawn uniformly over the valid space."""
    sizes = np.asarray([space_size(layout) for layout in layouts], dtype=float)
    if count > sizes.sum() - len(exclude):
        raise ValueError("not enough unseen valid schedules")
    probabilities = sizes / sizes.sum()
    rng = np.random.default_rng(seed)
    seen = set(exclude)
    samples = []
    while len(samples) < count:
        sketch_id = int(rng.choice(len(layouts), p=probabilities))
        position = random_position(rng, layouts[sketch_id]).astype(int)
        key = (sketch_id, tuple(int(value) for value in position))
        if key in seen:
            continue
        seen.add(key)
        samples.append((sketch_id, position))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "jetson_nano_confirm" / "source")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    original = pd.read_parquet(SOURCE / "pool.parquet", columns=["id", "sketch_id", "vector"])
    exclude = {(int(s), tuple(int(v) for v in vector)) for s, vector in zip(original.sketch_id, original.vector)}
    layouts = load_layouts(SOURCE)
    samples = sample_new(layouts, exclude, args.n, args.seed)

    first_id = int(original["id"].max()) + 1
    ids = list(range(first_id, first_id + len(samples)))
    pool = pd.DataFrame(
        {"id": ids, "sketch_id": [s for s, _ in samples], "vector": [p.tolist() for _, p in samples]}
    )
    order = np.random.default_rng(args.seed + 1).permutation(ids).tolist()

    args.output.mkdir(parents=True)
    pool.to_parquet(args.output / "pool.parquet", index=False)
    save_json(args.output / "splits.json", {
        "seed": args.seed,
        "train": [],
        "validation": [],
        "test": [int(value) for value in order],
        "nested_train": {},
        "note": "confirmation TEST set: new schedules, none in data/jetson_nano/pool.parquet",
    })
    for path in sorted(SOURCE.glob("layout_*.json")):
        shutil.copy2(path, args.output / path.name)
    counts = pool["sketch_id"].value_counts().sort_index().to_dict()
    print(f"Wrote {len(pool)} new schedules (ids {ids[0]}-{ids[-1]}, per sketch {counts}) to {args.output}")


if __name__ == "__main__":
    main()
