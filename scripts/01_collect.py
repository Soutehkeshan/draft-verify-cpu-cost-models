
#!/usr/bin/env python3
"""Collect a reproducible pool of measured CPU schedules."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tvm
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.encoding import describe_space, random_position
from dvcm.harness import HarnessConfig, make_matmul, measure
from dvcm.study import DATA_DIR, RESULTS_DIR, add_schedule_if_new, save_json

SEED = 0
N_TARGET = 5000
BATCH_SIZE = 250
COOLDOWN_S = 60
PILOT_N = 50
MAX_PROJECTED_HOURS = 8
CHECKPOINT_SIZE = 25


def _mac_value(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _power_state() -> str:
    output = _mac_value(["pmset", "-g", "batt"])
    return output.replace("\n", "; ")[:300]


def _json_layout(layout: dict, sketch_id: int) -> dict:
    return {
        "sketch_id": sketch_id,
        "tiles": [
            {
                "n": int(tile["n"]),
                "extent": int(tile["extent"]),
                "total_exp": int(tile["total_exp"]),
                "max_inner_exp": int(tile["max_inner_exp"]),
                "max_inner_factor": int(tile["max_inner_factor"]),
                "slice": [int(tile["slice"].start), int(tile["slice"].stop)],
            }
            for tile in layout["tiles"]
        ],
        "categoricals": [
            {
                "index": int(category["index"]),
                "candidates": [int(candidate) for candidate in category["candidates"]],
            }
            for category in layout["categoricals"]
        ],
        "dim": int(layout["dim"]),
        "parallel_tile_levels": {
            "i": int(layout["parallel_tile_levels"]["i"]),
            "j": int(layout["parallel_tile_levels"]["j"]),
        },
        "vector_lanes": 4,
        "num_cores": 4,
        "vector_registers": 32,
    }


def _append_rows(rows: list[dict], pool_path: Path) -> None:
    if not rows:
        return
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame(rows)
    if pool_path.exists():
        existing = pd.read_parquet(pool_path)
        incoming = pd.concat([existing, incoming], ignore_index=True)
        incoming = incoming.drop_duplicates(subset=["id"], keep="last")
    temporary_path = pool_path.with_suffix(pool_path.suffix + ".tmp")
    incoming.to_parquet(temporary_path, index=False)
    temporary_path.replace(pool_path)


def _sample_pool(sketches: list, layouts: list[dict], count: int, rng: np.random.Generator) -> list:
    counts = np.asarray(
        [
            np.prod(
                [len(tile["valid_exponents"]) for tile in layout["tiles"]]
                + [len(category["candidates"]) for category in layout["categoricals"]]
            )
            for layout in layouts
        ],
        dtype=float,
    )
    if count > int(counts.sum()):
        raise ValueError(
            f"Requested {count} unique schedules, but only {int(counts.sum())} are valid"
        )
    probabilities = counts / counts.sum()
    print(f"Valid schedules per sketch: {[int(value) for value in counts]} (weighted total={int(counts.sum())})")
    seen: set[tuple[int, tuple[int, ...]]] = set()
    samples = []
    while len(samples) < count:
        sketch_id = int(rng.choice(len(sketches), p=probabilities))
        position = random_position(rng, layouts[sketch_id]).astype(int)
        if not add_schedule_if_new(seen, sketch_id, position):
            continue
        samples.append((sketch_id, position))
    return samples


def _record(
    sample_id: int,
    sketch_id: int,
    position: np.ndarray,
    result,
    build_time_s: float,
    batch_id: int,
) -> dict:
    return {
        "id": int(sample_id),
        "sketch_id": int(sketch_id),
        "vector": [int(value) for value in position],
        "repeats_ms": list(result.repeats_ms),
        "median_ms": result.median_ms,
        "std_ms": result.std_ms,
        "status": result.status,
        "build_time_s": float(build_time_s),
        "batch_id": int(batch_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_TARGET)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    n_target = 30 if args.smoke else args.n
    repeats = 2 if args.smoke else 5
    cooldown = 0 if args.smoke else COOLDOWN_S
    if args.pilot_only and args.smoke:
        parser.error("--pilot-only cannot be combined with --smoke")
    if n_target < 1:
        parser.error("--n must be positive")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DATA_DIR / "smoke"
        if args.smoke
        else DATA_DIR
    )
    meta_path = output_dir / "run_meta.json"
    reduced_for_projection = False
    if not args.smoke and args.n == N_TARGET and meta_path.exists():
        with meta_path.open(encoding="utf-8") as source:
            previous_meta = json.load(source)
        if previous_meta.get("target_reduced_for_projection"):
            n_target = 3000
            reduced_for_projection = True
            print("Resuming the previously reduced 3,000-schedule target")
    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = tvm.target.Target(
        {"kind": "llvm", "mcpu": "apple-m2", "num-cores": 4}
    )
    context = ms.TuneContext(
        mod=make_matmul(512),
        target=target,
        space_generator=PostOrderApply(sch_rules="llvm", postprocs="llvm"),
        rand_state=SEED,
        num_threads=4,
    )
    sketches = context.generate_design_space()
    layouts = [describe_space(schedule.trace) for schedule in sketches]
    print(f"Design-space sketches: {len(sketches)}")
    for sketch_id, layout in enumerate(layouts):
        save_json(output_dir / f"layout_{sketch_id}.json", _json_layout(layout, sketch_id))

    rng = np.random.default_rng(SEED)
    sample_count = n_target if args.smoke else max(N_TARGET, args.n)
    samples = _sample_pool(sketches, layouts, sample_count, rng)
    run_order = rng.permutation(sample_count).tolist()
    canaries = [
        {"sketch_id": sketch_id, "vector": position.tolist()}
        for sketch_id, position in samples[:20]
    ]
    canary_path = output_dir / "canary.json"
    if not canary_path.exists():
        save_json(canary_path, canaries)
    else:
        with canary_path.open(encoding="utf-8") as source:
            canaries = json.load(source)

    pool_path = output_dir / "pool.parquet"
    existing = pd.read_parquet(pool_path) if pool_path.exists() else pd.DataFrame()
    measured_ids = set(existing["id"].astype(int)) if not existing.empty else set()
    remaining = [
        (idx, samples[idx])
        for idx in run_order[:n_target]
        if idx not in measured_ids
    ]
    if not remaining and meta_path.exists():
        statuses = Counter(existing["status"])
        print(f"Collection already complete: {len(existing)} measured schedules")
        print(
            "Status counts:",
            {
                name: int(statuses.get(name, 0))
                for name in ("ok", "build_fail", "wrong_result", "slow_skip")
            },
        )
        return
    start_time = datetime.now(timezone.utc)
    run_started = time.perf_counter()
    config = HarnessConfig(n=512, target=target, repeats=repeats, seed=SEED, slow_threshold_ms=float("inf"))

    pilot_count = min(PILOT_N, len(remaining))
    pending_rows = []
    pilot_sample_ids = []
    pilot_ms = []
    pilot_build = []
    for sample_id, (sketch_id, position) in remaining[:pilot_count]:
        result, build_s = measure(position, sketches[sketch_id], layouts[sketch_id], config)
        pending_rows.append(_record(sample_id, sketch_id, position, result, build_s, -1))
        pilot_sample_ids.append(sample_id)
        if len(pending_rows) >= 10:
            _append_rows(pending_rows, pool_path)
            measured_ids.update(int(row["id"]) for row in pending_rows)
            pending_rows.clear()
        if result.repeats_ms:
            pilot_ms.append(result.median_ms)
        pilot_build.append(build_s)
    if pending_rows:
        _append_rows(pending_rows, pool_path)
        measured_ids.update(int(row["id"]) for row in pending_rows)
        pending_rows.clear()
    pilot_frame = pd.read_parquet(pool_path)
    pilot_frame = pilot_frame[pilot_frame["id"].isin(pilot_sample_ids)]
    pilot_statuses = Counter(pilot_frame["status"])
    pilot_median = float(np.median(pilot_ms)) if pilot_ms else 1.0
    slow_threshold_ms = max(pilot_median * 20.0, 0.001)
    config = HarnessConfig(
        n=512,
        target=target,
        repeats=repeats,
        seed=SEED,
        slow_threshold_ms=slow_threshold_ms,
    )
    projection_build = float(np.mean(pilot_build[:20])) if pilot_build else 0.0
    projection_measure = float(np.mean(pilot_ms[:20])) / 1000.0 if pilot_ms else 0.0
    batch_count = (n_target + BATCH_SIZE - 1) // BATCH_SIZE
    projected_seconds = (
        (n_target + batch_count * len(canaries))
        * (projection_build + projection_measure)
        + max(0, batch_count - 1) * cooldown
    )
    print(f"Pilot median: {pilot_median:.6f} ms; slow threshold: {slow_threshold_ms:.6f} ms")
    print(
        "Pilot status counts:",
        {
            name: int(pilot_statuses.get(name, 0))
            for name in ("ok", "build_fail", "wrong_result", "slow_skip")
        },
    )
    print(f"Projected run time: {projected_seconds / 3600:.2f} hours")
    if not args.smoke and projected_seconds > MAX_PROJECTED_HOURS * 3600 and args.n == N_TARGET:
        n_target = 3000
        reduced_for_projection = True
        batch_count = (n_target + BATCH_SIZE - 1) // BATCH_SIZE
        projected_seconds = (
            (n_target + batch_count * len(canaries))
            * (projection_build + projection_measure)
            + max(0, batch_count - 1) * cooldown
        )
        print("Projection exceeds 8 hours; reducing N_TARGET to 3000")
        print(f"Reduced-pool projection: {projected_seconds / 3600:.2f} hours")

    if args.pilot_only:
        print("Pilot complete; collection was not started.")
        return

    canary_rows = []
    last_active_batch = None
    completed_count = len(set(run_order[:n_target]).intersection(measured_ids))
    run_samples_completed = 0
    for batch_id, start in enumerate(range(0, n_target, BATCH_SIZE)):
        batch_order = run_order[start : min(start + BATCH_SIZE, n_target)]
        batch_indexes = [
            sample_id
            for sample_id in batch_order
            if sample_id not in measured_ids
        ]
        if not batch_indexes:
            continue
        if last_active_batch is not None and cooldown:
            time.sleep(cooldown)
        batch_canary_rows = []
        for canary_id, item in enumerate(canaries):
            sketch_id = int(item["sketch_id"])
            position = np.asarray(item["vector"], dtype=int)
            result, build_s = measure(position, sketches[sketch_id], layouts[sketch_id], config)
            batch_canary_rows.append(
                {
                    **_record(-1, sketch_id, position, result, build_s, batch_id),
                    "canary_id": canary_id,
                }
            )
        canary_rows.extend(batch_canary_rows)
        last_active_batch = batch_id

        for sample_id in batch_indexes:
            sketch_id, position = samples[sample_id]
            result, build_s = measure(
                position, sketches[sketch_id], layouts[sketch_id], config
            )
            row = _record(sample_id, sketch_id, position, result, build_s, batch_id)
            pending_rows.append(row)
            if len(pending_rows) >= CHECKPOINT_SIZE:
                _append_rows(pending_rows, pool_path)
                pending_rows.clear()
            completed_count += 1
            run_samples_completed += 1
            completed = completed_count
            if completed >= 100 and completed % 25 == 0:
                elapsed = time.perf_counter() - run_started
                rate = run_samples_completed / max(elapsed, 1e-9) * 60
                eta_minutes = (n_target - completed) / max(rate, 1e-9)
                print(f"{completed} samples; {rate:.1f} samples/min; ETA {eta_minutes:.1f} min")
    _append_rows(pending_rows, pool_path)
    if canary_rows:
        canary_log_path = (
            output_dir / "canary_log.csv"
            if args.smoke
            else RESULTS_DIR / "canary.csv"
        )
        canary_frame = pd.DataFrame(canary_rows)
        if canary_log_path.exists():
            canary_frame = pd.concat(
                [pd.read_csv(canary_log_path), canary_frame], ignore_index=True
            )
        canary_frame.to_csv(canary_log_path, index=False)

    pool = pd.read_parquet(pool_path)
    statuses = Counter(pool["status"])
    ended = datetime.now(timezone.utc)
    meta = {
        "tvm_version": tvm.__version__,
        "target": str(target),
        "machine_model": _mac_value(["sysctl", "-n", "hw.model"]),
        "macos_version": platform.mac_ver()[0],
        "python_version": platform.python_version(),
        "power_state": _power_state(),
        "thread_settings": {"TVM_NUM_THREADS": "4", "target_num_cores": 4},
        "config": {
            "n": 512,
            "seed": SEED,
            "repeats": repeats,
            "number": 1,
            "min_repeat_ms": 10,
            "n_target": n_target,
            "batch_size": BATCH_SIZE,
            "cooldown_s": cooldown,
        },
        "collection_complete": completed_count >= n_target,
        "target_reduced_for_projection": reduced_for_projection,
        "slow_threshold_ms": slow_threshold_ms,
        "slow_skip_fraction": float((pool["status"] == "slow_skip").mean()),
        "start_timestamp": start_time.isoformat(),
        "end_timestamp": ended.isoformat(),
        "status_counts": {name: int(statuses.get(name, 0)) for name in ("ok", "build_fail", "wrong_result", "slow_skip")},
        "projection_seconds": projected_seconds,
    }
    save_json(meta_path, meta)
    print("Status counts:", meta["status_counts"])
    print(f"Slow-skip fraction: {meta['slow_skip_fraction']:.1%}")


if __name__ == "__main__":
    main()
