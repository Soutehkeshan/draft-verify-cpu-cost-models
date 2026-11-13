"""Shared repository paths and measured-pool loading helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
POOL_PATH = DATA_DIR / "pool.parquet"
# Compatibility for layouts collected before parallel-loop metadata was serialized.
_LEGACY_PARALLEL_TILE_LEVELS = {
    0: {"i": 2, "j": 2},
    1: {"i": 1, "j": 1},
    2: {"i": 2, "j": 2},
}


def load_layouts(
    data_dir: Path = DATA_DIR,
    target: Any | None = None,
) -> dict[int, dict[str, Any]]:
    """Load per-sketch layouts and bind features to the selected target."""
    layouts = {}
    for path in sorted(data_dir.glob("layout_*.json")):
        sketch_id = int(path.stem.removeprefix("layout_"))
        with path.open(encoding="utf-8") as source:
            layout = json.load(source)
        if "parallel_tile_levels" not in layout:
            try:
                layout["parallel_tile_levels"] = _LEGACY_PARALLEL_TILE_LEVELS[sketch_id]
            except KeyError as error:
                raise ValueError(
                    f"Legacy layout {path} lacks parallel-loop metadata"
                ) from error
        if target is not None:
            target_fields = {
                "vector_lanes": target.simd_width_f32,
                "num_cores": target.num_cores,
                "vector_registers": target.vector_registers,
            }
            for name, value in target_fields.items():
                if value is None:
                    layout.pop(name, None)
                else:
                    layout[name] = int(value)
        layouts[sketch_id] = layout
    if not layouts:
        raise FileNotFoundError(f"No layout_*.json files found in {data_dir}")
    return layouts


def load_pool(
    include_failures: bool = False,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Read the measured pool, retaining only modelable rows by default."""
    pool_path = data_dir / "pool.parquet"
    if not pool_path.exists():
        raise FileNotFoundError(f"Measured data not found: {pool_path}; run make data")
    frame = pd.read_parquet(pool_path)
    if not include_failures:
        frame = frame[
            frame["status"].isin(("ok", "slow_skip"))
            & np.isfinite(frame["median_ms"])
            & (frame["median_ms"] > 0)
        ].copy()
    return frame.reset_index(drop=True)


def positions_and_features(
    frame: pd.DataFrame,
    include_derived: bool = True,
    data_dir: Path = DATA_DIR,
    target: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert table records to encoded position and feature matrices."""
    from dvcm.features import features

    layouts = load_layouts(data_dir, target)
    positions = np.asarray([np.asarray(row, dtype=float) for row in frame["vector"]])
    matrix = np.asarray(
        [
            features(position, int(sketch_id), layouts[int(sketch_id)], include_derived)
            for position, sketch_id in zip(positions, frame["sketch_id"])
        ],
        dtype=float,
    )
    return positions, matrix


def load_splits(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Load deterministic schedule-ID splits from a selected dataset."""
    path = data_dir / "splits.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}; run make models")
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def make_splits(frame: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """Create reproducible per-sketch 70/15/15 splits and nested TRAIN subsets."""
    rng = np.random.default_rng(seed)
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
    nested = {
        str(size): subset_order[: min(size, len(subset_order))].tolist()
        for size in (100, 250, 500, 1000, 2000)
    }
    nested["all"] = subset_order.tolist()
    return {
        "seed": seed,
        "train": sorted(train),
        "validation": sorted(validation),
        "test": sorted(test),
        "nested_train": nested,
    }


def validate_splits(splits: dict[str, Any], sample_ids: set[int]) -> None:
    """Require disjoint, exhaustive schedule splits and nested TRAIN subsets."""
    partition_names = ("train", "validation", "test")
    partitions = {
        name: {int(value) for value in splits[name]}
        for name in partition_names
    }
    for index, name in enumerate(partition_names):
        for other in partition_names[index + 1 :]:
            overlap = partitions[name] & partitions[other]
            if overlap:
                raise ValueError(f"{name} and {other} overlap at schedule IDs {sorted(overlap)[:5]}")
    combined = set().union(*partitions.values())
    if combined != sample_ids:
        missing = sorted(sample_ids - combined)[:5]
        extra = sorted(combined - sample_ids)[:5]
        raise ValueError(f"Split IDs do not match samples; missing={missing}, extra={extra}")

    train_ids = partitions["train"]
    nested = {
        name: {int(value) for value in values}
        for name, values in splits["nested_train"].items()
    }
    if any(not ids <= train_ids for ids in nested.values()):
        raise ValueError("Nested training subsets must be contained in TRAIN")
    ordered = sorted(
        ((len(ids), name, ids) for name, ids in nested.items() if name != "all"),
        key=lambda item: item[0],
    )
    previous: set[int] = set()
    for _, name, ids in ordered:
        if not previous <= ids:
            raise ValueError(f"Nested training subset {name} is not nested")
        previous = ids
    if "all" in nested and nested["all"] != train_ids:
        raise ValueError("The 'all' nested training subset must equal TRAIN")


def save_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def add_schedule_if_new(
    seen: set[tuple[int, tuple[int, ...]]], sketch_id: int, vector: list[int] | np.ndarray
) -> bool:
    """Remember a sketch/vector key once; return False for repeated schedules."""
    key = (int(sketch_id), tuple(int(value) for value in vector))
    if key in seen:
        return False
    seen.add(key)
    return True
