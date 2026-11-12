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
from dvcm.provenance import capture_provenance, write_csv
from dvcm.study import DATA_DIR, RESULTS_DIR, load_layouts, load_pool, load_splits

EXPECTED_SIGNS = {
    "bytes_moved": 1,
    "vector_lane_utilisation": -1,
    "parallel_utilisation": -1,
    "register_utilisation": -1,
}
FLAT_RHO_THRESHOLD = 0.02


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    provenance = capture_provenance(ROOT, allow_dirty=args.allow_dirty)

    metadata_path = DATA_DIR / "run_meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Collection metadata not found: {metadata_path}")
    import json

    with metadata_path.open(encoding="utf-8") as source:
        metadata = json.load(source)
    if not metadata.get("collection_complete", False):
        raise RuntimeError("Feature correlations require a completed collection")

    frame = load_pool()
    splits = load_splits()
    train_ids = {int(value) for value in splits["train"]}
    train = frame[frame["id"].astype(int).isin(train_ids)]
    if train.empty:
        raise ValueError("The TRAIN split has no modelable measurements")

    layouts = load_layouts()
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "feature_correlations.csv"
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
