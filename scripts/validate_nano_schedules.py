#!/usr/bin/env python3
"""Qualify a bounded sample of Nano MetaSchedule candidates over RPC."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import tvm
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.config import load_target_config, load_workload_config
from dvcm.encoding import decisions_to_position, describe_space, random_position
from dvcm.harness import HarnessConfig, MeasurementResult, make_matmul, measure
from dvcm.rpc import RPCExecutor

SAMPLES_PER_SKETCH = 16
STABILITY_REPEATS = 7
STABILITY_MIN_REPEAT_MS = 20
STABILITY_BATCHES = 3


def _sample_positions(
    base_position: np.ndarray,
    layout: dict,
    rng: np.random.Generator,
    count: int,
) -> list[tuple[str, np.ndarray]]:
    """Cover categorical choices and tile-choice edges before random fills."""
    if count < 1:
        raise ValueError("count must be positive")
    if count > _space_size(layout):
        raise ValueError("count exceeds the valid schedule-space size")
    selected: dict[tuple[int, ...], str] = {}

    def add(position: np.ndarray, label: str) -> None:
        selected.setdefault(tuple(int(value) for value in position), label)

    base = np.asarray(base_position, dtype=int)
    add(base, "recorded-default")

    for categorical in layout["categoricals"]:
        for choice in range(len(categorical["candidates"])):
            position = base.copy()
            position[categorical["index"]] = choice
            add(position, f"category-{choice}")

    for tile_index, tile in enumerate(layout["tiles"]):
        choices = tile["valid_exponents"]
        if not choices:
            raise ValueError("tile layout has no valid exponent choices")
        for edge_index in sorted({0, len(choices) // 2, len(choices) - 1}):
            position = base.copy()
            position[tile["slice"]] = choices[edge_index]
            add(position, f"tile-{tile_index}-choice-{edge_index}")

    while len(selected) < count:
        add(random_position(rng, layout).astype(int), f"seeded-random-{len(selected)}")

    return [
        (label, np.asarray(position, dtype=int))
        for position, label in list(selected.items())[:count]
    ]


def _space_size(layout: dict) -> int:
    return math.prod(
        [len(tile["valid_exponents"]) for tile in layout["tiles"]]
        + [len(category["candidates"]) for category in layout["categoricals"]]
    )


def _measurement_row(
    sketch_id: int,
    label: str,
    position: np.ndarray,
    result: MeasurementResult,
    timing_batch: int | None,
) -> dict:
    timing = result.timing
    median_ms = timing.median_ms
    return {
        "sketch_id": sketch_id,
        "sample": label,
        "timing_batch": timing_batch,
        "position": [int(value) for value in position],
        "status": timing.status,
        "build_time_s": result.build_time_s,
        "export_time_s": result.export_time_s,
        "upload_time_s": result.upload_time_s,
        "link_load_time_s": result.link_load_time_s,
        "repeats_ms": list(timing.repeats_ms),
        "median_ms": median_ms,
        "std_ms": timing.std_ms,
        "repeat_cv_pct": (
            100.0 * timing.std_ms / median_ms
            if timing.repeats_ms and median_ms > 0
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")

    target_config = load_target_config("jetson_nano", ROOT)
    workload = load_workload_config(root=ROOT)
    target = target_config.make_tvm_target()
    executor = RPCExecutor.connect(args.host, args.port)

    context = ms.TuneContext(
        mod=make_matmul(workload.n),
        target=target,
        space_generator=PostOrderApply(sch_rules="llvm", postprocs="llvm"),
        rand_state=workload.seed,
        num_threads=target_config.thread_settings.get("TVM_NUM_THREADS", 1),
    )
    sketches = context.generate_design_space()
    if not sketches:
        raise RuntimeError("MetaSchedule produced no candidate sketches")
    layouts = [describe_space(sketch.trace) for sketch in sketches]
    config = HarnessConfig(
        target=target,
        postprocessors="llvm",
        n=workload.n,
        repeats=STABILITY_REPEATS,
        number=1,
        min_repeat_ms=STABILITY_MIN_REPEAT_MS,
        seed=workload.seed,
        slow_threshold_ms=float("inf"),
    )

    rows = []
    coverage = []
    for sketch_id, (sketch, layout) in enumerate(zip(sketches, layouts)):
        base = decisions_to_position(sketch.trace, layout).astype(int)
        candidates = _sample_positions(
            base,
            layout,
            np.random.default_rng(workload.seed + sketch_id),
            SAMPLES_PER_SKETCH,
        )
        coverage.append(
            {
                "sketch_id": sketch_id,
                "valid_positions": _space_size(layout),
                "sampled_positions": len(candidates),
                "tile_count": len(layout["tiles"]),
                "categorical_choices": [
                    len(category["candidates"])
                    for category in layout["categoricals"]
                ],
            }
        )
        for label, position in candidates:
            is_default = label == "recorded-default"
            batches = range(STABILITY_BATCHES) if is_default else (None,)
            for timing_batch in batches:
                candidate_config = (
                    config
                    if is_default
                    else HarnessConfig(
                        target=target,
                        postprocessors="llvm",
                        n=workload.n,
                        repeats=1,
                        number=1,
                        min_repeat_ms=0,
                        seed=workload.seed,
                        slow_threshold_ms=float("inf"),
                    )
                )
                result = measure(
                    position,
                    sketch,
                    layout,
                    candidate_config,
                    rpc_executor=executor,
                )
                rows.append(
                    _measurement_row(
                        sketch_id, label, position, result, timing_batch
                    )
                )

    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    stability = []
    for sketch_id in range(len(sketches)):
        batches = [
            row
            for row in rows
            if row["sketch_id"] == sketch_id
            and row["sample"] == "recorded-default"
        ]
        values = [value for row in batches for value in row["repeats_ms"]]
        median_ms = float(np.median(values)) if values else float("nan")
        std_ms = float(np.std(values)) if values else float("nan")
        stability.append(
            {
                "sketch_id": sketch_id,
                "timing_batches": len(batches),
                "repeats_per_batch": STABILITY_REPEATS,
                "batch_medians_ms": [row["median_ms"] for row in batches],
                "combined_median_ms": median_ms,
                "combined_std_ms": std_ms,
                "combined_cv_pct": (
                    100.0 * std_ms / median_ms if median_ms > 0 else None
                ),
                "min_ms": min(values) if values else None,
                "max_ms": max(values) if values else None,
            }
        )
    report = {
        "collection_started": False,
        "dataset_written": False,
        "unique_candidate_count": sum(
            item["sampled_positions"] for item in coverage
        ),
        "measurement_count": len(rows),
        "target_id": target_config.target_id,
        "target_profile": target_config.to_dict(),
        "target": str(target),
        "execution_backend": "rpc",
        "rpc_host": args.host,
        "rpc_port": args.port,
        "tvm_version": str(tvm.__version__),
        "workload": workload.to_dict(),
        "schedule_rules": "llvm",
        "postprocessors": "llvm",
        "stability_batches": STABILITY_BATCHES,
        "stability_repeats": STABILITY_REPEATS,
        "stability_min_repeat_ms": STABILITY_MIN_REPEAT_MS,
        "input_prepare_time_s": executor.input_prepare_time_s,
        "input_upload_time_s": executor.input_upload_time_s,
        "coverage": coverage,
        "stability": stability,
        "status_counts": statuses,
        "candidates": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if statuses == {"ok": len(rows)} else 1


if __name__ == "__main__":
    raise SystemExit(main())
