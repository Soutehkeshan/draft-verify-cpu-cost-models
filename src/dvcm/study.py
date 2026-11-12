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


def load_layouts() -> dict[int, dict[str, Any]]:
    """Load serializable per-sketch layouts."""
    layouts = {}
    for path in sorted(DATA_DIR.glob("layout_*.json")):
        sketch_id = int(path.stem.removeprefix("layout_"))
        with path.open(encoding="utf-8") as source:
            layouts[sketch_id] = json.load(source)
    if not layouts:
        raise FileNotFoundError(f"No layout_*.json files found in {DATA_DIR}")
    return layouts


def load_pool(include_failures: bool = False) -> pd.DataFrame:
    """Read the measured pool, retaining only modelable rows by default."""
    if not POOL_PATH.exists():
        raise FileNotFoundError(f"Measured data not found: {POOL_PATH}; run make data")
    frame = pd.read_parquet(POOL_PATH)
    if not include_failures:
        frame = frame[
            frame["status"].isin(("ok", "slow_skip"))
            & np.isfinite(frame["median_ms"])
            & (frame["median_ms"] > 0)
        ].copy()
    return frame.reset_index(drop=True)


def positions_and_features(
    frame: pd.DataFrame, include_derived: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Convert table records to encoded position and feature matrices."""
    from dvcm.features import features

    layouts = load_layouts()
    positions = np.asarray([np.asarray(row, dtype=float) for row in frame["vector"]])
    matrix = np.asarray(
        [
            features(position, int(sketch_id), layouts[int(sketch_id)], include_derived)
            for position, sketch_id in zip(positions, frame["sketch_id"])
        ],
        dtype=float,
    )
    return positions, matrix


def load_splits() -> dict[str, Any]:
    """Load deterministic schedule-ID splits."""
    path = DATA_DIR / "splits.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}; run make models")
    with path.open(encoding="utf-8") as source:
        return json.load(source)


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
