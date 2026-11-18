"""Shared repository paths and measured-pool loading helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "jetson_nano"


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
            raise ValueError(f"Layout {path} lacks parallel-loop metadata")
        if target is not None:
            from dvcm.features import bind_target_settings

            layout = bind_target_settings(layout, target)
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


def load_splits(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Load deterministic schedule-ID splits from a selected dataset."""
    path = data_dir / "splits.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}; run make models")
    with path.open(encoding="utf-8") as source:
        return json.load(source)


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


def collection_complete(metadata: dict[str, Any]) -> bool:
    """Read the completion flag of a collection's run_meta.json."""
    return bool((metadata.get("collection") or {}).get("collection_complete", False))


def save_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


