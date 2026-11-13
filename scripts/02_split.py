
#!/usr/bin/env python3
"""Create deterministic stratified train/validation/test schedule splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.config import (
    add_target_argument,
    experiment_metadata,
    load_target_config,
    load_workload_config,
)
from dvcm.study import load_pool, load_splits, make_splits, save_json, validate_splits


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
    splits = make_splits(frame, seed=workload.seed)
    validate_splits(splits, set(frame["id"].astype(int)))
    context = experiment_metadata(target, workload, data_dir)
    splits.update(
        {
            "target_id": target.target_id,
            "workload_id": workload.workload_id,
            "pool_sha256": context["dataset"].get("pool.parquet_sha256"),
        }
    )
    save_json(split_path, splits)
    print(
        f"Split {len(frame)} valid samples: "
        f"train={len(splits['train'])}, validation={len(splits['validation'])}, "
        f"test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
