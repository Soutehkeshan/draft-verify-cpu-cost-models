#!/usr/bin/env python3
"""Report derived-feature Spearman correlations on TRAIN only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.features import ALL_DERIVED_NAMES, DERIVED_NAMES, derived
from dvcm.config import (
    add_target_argument,
    experiment_metadata,
    load_target_config,
    load_workload_config,
)
from dvcm.provenance import capture_provenance, write_csv
from dvcm.study import collection_complete, load_layouts, load_pool, load_splits

EXPECTED_SIGNS = {
    "bytes_moved": 1,
    "vector_lane_utilisation": -1,
    "parallel_utilisation": -1,
    "register_utilisation": -1,
}
FLAT_RHO_THRESHOLD = 0.02


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

    metadata_path = data_dir / "run_meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Collection metadata not found: {metadata_path}")
    import json

    with metadata_path.open(encoding="utf-8") as source:
        metadata = json.load(source)
    if metadata.get("target_id", target.target_id) != target.target_id:
        raise ValueError("Selected target does not match the collected dataset")
    if not collection_complete(metadata):
        raise RuntimeError("Feature correlations require a completed collection")

    frame = load_pool(data_dir=data_dir)
    splits = load_splits(data_dir)
    train_ids = {int(value) for value in splits["train"]}
    train = frame[frame["id"].astype(int).isin(train_ids)]
    if train.empty:
        raise ValueError("The TRAIN split has no modelable measurements")

    layouts = load_layouts(data_dir, target)
    values_by_name = {name: [] for name in ALL_DERIVED_NAMES}
    for row in train.itertuples(index=False):
        position = np.asarray(row.vector, dtype=float)
        descriptors = derived(position, layouts[int(row.sketch_id)])
        for name in ALL_DERIVED_NAMES:
            values_by_name[name].append(descriptors[name])
    latency = train["median_ms"].to_numpy(dtype=float)

    results = []
    for name in ALL_DERIVED_NAMES:
        values = np.asarray(values_by_name[name], dtype=float)
        unique_values = int(np.unique(values).size)
        if unique_values < 2 or np.unique(latency).size < 2:
            rho = float("nan")
            p_value = float("nan")
            status = "flat"
        else:
            result = spearmanr(values, latency)
            rho = float(result.statistic)
            p_value = float(result.pvalue)
            expected_sign = EXPECTED_SIGNS.get(name)
            status = "flat" if abs(rho) < FLAT_RHO_THRESHOLD else "ok"
            if expected_sign is not None and rho * expected_sign < 0:
                status = "rethink"
        results.append(
            {
                "target_id": target.target_id,
                "feature": name,
                "train_rows": len(train),
                "unique_values": unique_values,
                "spearman_rho": rho,
                "p_value": p_value,
                "expected_direction": {
                    1: "positive",
                    -1: "negative",
                    None: "unspecified",
                }[EXPECTED_SIGNS.get(name)],
                "included_in_model": name in DERIVED_NAMES,
                "status": status,
            }
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "feature_correlations.csv"
    write_csv(pd.DataFrame(results), output_path, provenance)
    print(f"TRAIN-only Spearman correlations ({len(train)} rows):")
    for result in results:
        print(
            f"{result['feature']}: rho={result['spearman_rho']:.4f}, "
            f"unique={result['unique_values']}, status={result['status']}"
        )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
