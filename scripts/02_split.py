
#!/usr/bin/env python3
"""Create deterministic stratified train/validation/test schedule splits.

A pool that re-measured another pool's positions (``01_collect.py
--positions-from``) reuses that pool's split restricted to its own modelable
IDs, so both targets are scored on the same TEST schedules. The source split
is the one recorded, with its sha256, in the collection's ``run_meta.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.config import (
    add_target_argument,
    experiment_metadata,
    load_target_config,
    load_workload_config,
)
from dvcm.study import (
    load_pool,
    load_splits,
    make_splits,
    restrict_splits,
    save_json,
    validate_splits,
)


def source_split(data_dir: Path, root: Path = ROOT) -> tuple[Path, str] | None:
    """Return the split a replayed pool's positions came from, checking its sha256."""
    meta_path = data_dir / "run_meta.json"
    if not meta_path.is_file():
        return None
    with meta_path.open(encoding="utf-8") as source:
        collection = json.load(source).get("collection") or {}
    if not collection.get("splits"):
        return None
    path = Path(collection["splits"])
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Source split recorded in {meta_path} not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != collection.get("splits_sha256"):
        raise ValueError(
            f"Source split {path} changed since the collection: sha256 {digest} "
            f"!= recorded {collection.get('splits_sha256')}"
        )
    return path, digest


def build_splits(
    frame: pd.DataFrame, data_dir: Path, seed: int, root: Path = ROOT
) -> dict[str, Any]:
    """Derive the split from the recorded source split, or draw a new one."""
    sample_ids = set(frame["id"].astype(int))
    source = source_split(data_dir, root)
    if source is None:
        splits = make_splits(frame, seed=seed)
    else:
        path, digest = source
        with path.open(encoding="utf-8") as handle:
            splits = restrict_splits(json.load(handle), sample_ids)
        try:
            shown = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            shown = str(path)
        splits["derived_from"] = {
            "path": shown,
            "sha256": digest,
            "rule": "source split restricted to this pool's modelable IDs; list order kept",
        }
    validate_splits(splits, sample_ids)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    add_target_argument(parser)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    target = load_target_config(args.target, ROOT)
    workload = load_workload_config(root=ROOT)
    data_dir, _ = target.paths(ROOT)
    frame = load_pool(data_dir=data_dir)
    split_path = data_dir / "splits.json"
    if split_path.exists() and not args.force:
        existing = load_splits(data_dir)
        validate_splits(existing, set(frame["id"].astype(int)))
        if existing.get("target_id", target.target_id) != target.target_id:
            raise ValueError("Existing split belongs to a different target")
        print(f"Preserving existing split: {split_path}")
        return
    splits = build_splits(frame, data_dir, workload.seed)
    context = experiment_metadata(target, workload, data_dir)
    splits.update(
        {
            "target_id": target.target_id,
            "workload_id": workload.workload_id,
            "pool_sha256": context["dataset"].get("pool.parquet_sha256"),
        }
    )
    save_json(split_path, splits)
    derived = splits.get("derived_from")
    print(
        f"Split {len(frame)} valid samples: "
        f"train={len(splits['train'])}, validation={len(splits['validation'])}, "
        f"test={len(splits['test'])}"
        + (f" (derived from {derived['path']}, sha256 {derived['sha256']})" if derived else "")
    )


if __name__ == "__main__":
    main()
