#!/usr/bin/env python3
"""Re-time the Nano TEST schedules once more, in one new RPC session.

Reads the Nano dataset (pool.parquet, splits.json, run_meta.json, canary.json,
layout_*.json) and writes next to it test_retime.parquet,
test_retime_canary.parquet, test_retime_batches.jsonl and
test_retime_meta.json. The inputs are only read; their checksums are recorded
before and after the run, and existing outputs are never overwritten.

Every TEST schedule is rebuilt and timed with --repeats repeats (default 15)
and the slow-skip shortcut disabled, in a seeded random order, in batches that
keep the collection's canaries, cooldowns, device-state probes and stop rules.
The run opens exactly one RPC session and never reconnects or resumes: an RPC
failure, canary drift, a frequency drop, a non-ok share above the collection
limit, or a server change stops it, leaving test_retime.partial.parquet and the
metadata for inspection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.config import load_target_config, load_workload_config
from dvcm.device_state import read_device_state
from dvcm.harness import HarnessConfig, measure
from dvcm.provenance import capture_provenance
from dvcm.rpc import RPCExecutor
from dvcm.study import save_json


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "dvcm_collect_for_retime", ROOT / "scripts" / "01_collect.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COLLECT = _load_collector()  # canary, drift, frequency and record helpers of the collection
STOP_EXIT_CODE = 3
ORDER_SEED = 0
OUTPUT_NAMES = {
    "pool": "test_retime.parquet",
    "partial": "test_retime.partial.parquet",
    "canary": "test_retime_canary.parquet",
    "batches": "test_retime_batches.jsonl",
    "meta": "test_retime_meta.json",
}
INPUT_PATTERNS = ("pool.parquet", "splits.json", "run_meta.json", "canary.json", "layout_*.json")


def measurement_order(test_ids: list[int], seed: int = ORDER_SEED) -> list[int]:
    """Seeded permutation of the stored TEST order (the collection used the stored order)."""
    ids = np.asarray([int(value) for value in test_ids], dtype=int)
    return [int(value) for value in np.random.default_rng(seed).permutation(ids)]


def read_server_environment(ssh_target: str, pid: int, timeout_s: float = 30.0) -> dict[str, str]:
    """Read the RPC server's environment from /proc/<pid>/environ on the Nano (read-only)."""
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh_target,
         "cat", f"/proc/{int(pid)}/environ"],
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"could not read /proc/{pid}/environ ({completed.returncode}): "
            f"{completed.stderr.decode(errors='replace').strip()[-300:]}"
        )
    text = completed.stdout.decode(errors="replace")
    return dict(item.split("=", 1) for item in text.split("\0") if "=" in item)


def check_server(state: dict[str, Any] | None, rpc_server: dict[str, Any]) -> dict[str, Any]:
    """Require the recorded RPC server process with its recorded TVM_/OMP_ environment."""
    pid = int(rpc_server["server_pid"])
    wanted = {
        name: value
        for name, value in rpc_server["environment"].items()
        if name.startswith(("TVM_", "OMP_"))
    }
    servers = (state or {}).get("rpc_servers", [])
    matching = [server for server in servers if int(server.get("pid", -1)) == pid]
    if not matching:
        raise RuntimeError(
            f"RPC server pid {pid} is not running (servers: {[s.get('pid') for s in servers]})"
        )
    if matching[0].get("environment") != wanted:
        raise RuntimeError(
            f"RPC server environment differs from run_meta.json: "
            f"{matching[0].get('environment')} != {wanted}"
        )
    return {"pid": pid, "environment": dict(matching[0]["environment"])}


def _probe(ssh_target: str) -> tuple[dict | None, list[str]]:
    try:
        return read_device_state(ssh_target), []
    except Exception as error:
        return None, [f"{type(error).__name__}: {error}"]


def _input_hashes(data_dir: Path) -> dict[str, str]:
    paths = sorted({path for pattern in INPUT_PATTERNS for path in data_dir.glob(pattern)})
    return {path.name: COLLECT._sha256(path) for path in paths}


def _verify_layouts(layouts: list[dict], data_dir: Path) -> None:
    files = sorted(data_dir.glob("layout_*.json"))
    if len(files) != len(layouts):
        raise SystemExit(
            f"Refusing to re-time: {len(files)} layout files but {len(layouts)} sketches"
        )
    for sketch_id, layout in enumerate(layouts):
        recorded = json.loads((data_dir / f"layout_{sketch_id}.json").read_text(encoding="utf-8"))
        if recorded != COLLECT.layout_to_json(layout, sketch_id):
            raise SystemExit(
                f"Refusing to re-time: sketch {sketch_id} design space differs from layout_{sketch_id}.json"
            )


def _write_parquet(rows: list[dict], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default="jetson_nano", choices=("jetson_nano",))
    parser.add_argument("--data-dir", type=Path, help="Nano dataset directory (default: data/nano)")
    parser.add_argument("--rpc-host", required=True)
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument("--ssh-target", required=True, help="Nano SSH target for the read-only probes")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=COLLECT.BATCH_SIZE)
    parser.add_argument("--cooldown-s", type=float, default=float(COLLECT.COOLDOWN_S))
    parser.add_argument("--order-seed", type=int, default=ORDER_SEED)
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="allow an uncommitted worktree (recorded in provenance; for tests)",
    )
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.batch_size < 1 or args.cooldown_s < 0:
        parser.error("--repeats and --batch-size must be positive and --cooldown-s non-negative")

    target_config = load_target_config(args.target, ROOT)
    workload = load_workload_config(root=ROOT)
    data_dir = (args.data_dir or target_config.paths(ROOT)[0]).resolve()
    outputs = {name: data_dir / filename for name, filename in OUTPUT_NAMES.items()}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite existing re-timing outputs: {existing}")

    # Refusals come before any device contact or write.
    run_meta = json.loads((data_dir / "run_meta.json").read_text(encoding="utf-8"))
    rpc_server = run_meta.get("rpc_server") or {}
    if not rpc_server.get("environment") or not rpc_server.get("server_pid"):
        raise SystemExit("Refusing to re-time: run_meta.json lacks the RPC server pid and environment")
    target_config.require_benchmarking_profile()
    provenance = capture_provenance(ROOT, allow_dirty=args.allow_dirty)
    inputs_before = _input_hashes(data_dir)

    pool = pd.read_parquet(data_dir / "pool.parquet", columns=["id", "sketch_id", "vector"])
    splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
    by_id = {
        int(row.id): (int(row.sketch_id), np.asarray(row.vector, dtype=int))
        for row in pool.itertuples()
    }
    test_ids = [int(value) for value in splits["test"]]
    missing = [value for value in test_ids if value not in by_id]
    if missing or len(set(test_ids)) != len(test_ids):
        raise SystemExit(f"Refusing to re-time: TEST ids missing from the pool or repeated: {missing[:5]}")
    order = measurement_order(test_ids, args.order_seed)
    position_of = {sample_id: index for index, sample_id in enumerate(order)}
    canaries = json.loads((data_dir / "canary.json").read_text(encoding="utf-8"))

    target = target_config.make_tvm_target()
    context = COLLECT.ms.TuneContext(
        mod=COLLECT.make_matmul(workload.n),
        target=target,
        space_generator=COLLECT.PostOrderApply(
            sch_rules=target_config.schedule_rules,
            postprocs=target_config.postprocessors,
        ),
        rand_state=workload.seed,
        num_threads=target_config.thread_settings.get("TVM_NUM_THREADS", 1),
    )
    sketches = context.generate_design_space()
    layouts = [COLLECT.describe_space(schedule.trace) for schedule in sketches]
    _verify_layouts(layouts, data_dir)
    for sample_id in order:
        sketch_id, position = by_id[sample_id]
        COLLECT._require_valid(position, layouts[sketch_id], f"id {sample_id}")
    for index, item in enumerate(canaries):
        COLLECT._require_valid(
            np.asarray(item["vector"], dtype=int), layouts[int(item["sketch_id"])], f"canary {index}"
        )
    baseline, baseline_source = COLLECT.canary_baseline(data_dir, canaries)
    config = HarnessConfig(
        target=target,
        postprocessors=target_config.postprocessors,
        n=workload.n,
        repeats=args.repeats,
        number=workload.measurement["number"],
        min_repeat_ms=workload.measurement["min_repeat_ms"],
        seed=workload.seed,
        slow_threshold_ms=math.inf,  # every TEST schedule gets the full repeat count
    )

    first_state, probe_errors = _probe(args.ssh_target)
    if first_state is None:
        raise SystemExit(f"Refusing to re-time: device-state probe failed: {probe_errors}")
    try:
        server = check_server(first_state, rpc_server)
        full_environment = read_server_environment(args.ssh_target, server["pid"])
    except RuntimeError as error:
        raise SystemExit(f"Refusing to re-time: {error}") from error
    observed = {name: full_environment.get(name) for name in rpc_server["environment"]}
    if observed != rpc_server["environment"]:
        raise SystemExit(
            "Refusing to re-time: RPC server environment differs from run_meta.json: "
            f"{observed} != {rpc_server['environment']}"
        )

    executor = RPCExecutor.connect(args.rpc_host, args.rpc_port)  # the only session of this run
    collection = run_meta.get("collection") or {}
    batches = [order[start : start + args.batch_size] for start in range(0, len(order), args.batch_size)]
    rows: list[dict] = []
    canary_rows: list[dict] = []
    status = {"state": "running", "stop_reason": None}
    started_utc = datetime.now(timezone.utc).isoformat()
    run_started = time.monotonic()

    def write_meta(inputs_after: dict[str, str] | None = None) -> None:
        save_json(
            outputs["meta"],
            {
                "description": "Second timing of the Nano TEST schedules; the collection's "
                "pool.parquet medians remain the primary result.",
                "state": status["state"],
                "stop_reason": status["stop_reason"],
                "target_id": target_config.target_id,
                "data_dir": str(data_dir),
                "outputs": dict(OUTPUT_NAMES),
                "n_target": len(order),
                "measured": len(rows),
                "status_counts": dict(Counter(row["status"] for row in rows)),
                "measurement_order": order,
                "order_seed": args.order_seed,
                "order_rule": "numpy default_rng(order_seed).permutation of splits.json test",
                "repeats": config.repeats,
                "number": config.number,
                "min_repeat_ms": config.min_repeat_ms,
                "slow_skip": "disabled",
                "collection_slow_threshold_ms": run_meta.get("slow_threshold_ms"),
                "batch_size": args.batch_size,
                "cooldown_s": args.cooldown_s,
                "canaries": len(canaries),
                "canary_baseline": baseline_source,
                "stop_rules": {
                    "canary_drift_pct": COLLECT.DRIFT_LIMIT_PCT,
                    "drift_batches": COLLECT.DRIFT_BATCHES,
                    "non_ok_fraction": COLLECT.NON_OK_LIMIT,
                    "frequency_margin": COLLECT.FREQUENCY_MARGIN,
                    "rpc_failure": "stop; no reconnect",
                    "server": "pid and TVM_/OMP_ environment must match run_meta.json before and after every batch",
                },
                "rpc_host": args.rpc_host,
                "rpc_port": args.rpc_port,
                "rpc_sessions_opened": 1,
                "reconnects": 0,
                "rpc_server_check": {
                    "pid": server["pid"],
                    "environment_checked": sorted(rpc_server["environment"]),
                    "matches_run_meta": True,
                },
                "rpc_input_prepare_time_s": executor.input_prepare_time_s,
                "rpc_input_upload_time_s": executor.input_upload_time_s,
                "original_collection": {
                    name: collection.get(name)
                    for name in ("repeats", "number", "min_repeat_ms", "git_commit",
                                 "rpc_sessions_opened", "start_timestamp", "last_update")
                },
                "protocol_differences": [
                    f"{config.repeats} repeats instead of {collection.get('repeats')}",
                    "slow-skip disabled, so every schedule gets every repeat",
                    "seeded random order instead of the stored TEST order",
                    "a new RPC session to the same server process",
                ],
                "inputs_sha256_before": inputs_before,
                "inputs_sha256_after": inputs_after,
                "start_timestamp": started_utc,
                "last_update": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": time.monotonic() - run_started,
                "provenance": provenance,
            },
        )

    def stop(reason: str) -> None:
        status.update(state="stopped", stop_reason=reason)
        if rows:
            _write_parquet(rows, outputs["partial"])
        write_meta()
        print(f"STOPPED: {reason}. No reconnect or resume; outputs kept for inspection.", flush=True)
        raise SystemExit(STOP_EXIT_CODE)

    print(
        f"Re-timing {len(order)} TEST schedules with {config.repeats} repeats in {len(batches)} "
        f"batches of {args.batch_size} (cooldown {args.cooldown_s:g} s, {len(canaries)} canaries/batch, "
        f"baseline: {baseline_source}); server pid {server['pid']}; commit {provenance['git_commit'][:12]}",
        flush=True,
    )
    write_meta()
    drifted_streak = 0
    for batch_id, batch_ids in enumerate(batches):
        if batch_id and args.cooldown_s:
            time.sleep(args.cooldown_s)
        batch_started = time.monotonic()
        batch_started_utc = datetime.now(timezone.utc).isoformat()
        before, errors = _probe(args.ssh_target)
        if before is None:
            stop(f"device-state probe failed before batch {batch_id}: {errors}")
        try:
            check_server(before, rpc_server)
        except RuntimeError as error:
            stop(str(error))
        batch_rows: list[dict] = []
        batch_canaries: list[dict] = []
        failure = None
        drift = None
        try:
            for canary_id, item in enumerate(canaries):
                sketch_id = int(item["sketch_id"])
                position = np.asarray(item["vector"], dtype=int)
                result = measure(position, sketches[sketch_id], layouts[sketch_id], config, executor)
                batch_canaries.append(
                    {
                        **COLLECT._record(-1, sketch_id, position, result, batch_id),
                        "canary_id": canary_id,
                        "source_id": int(item.get("id", -1)),
                    }
                )
            drift = COLLECT.canary_drift_pct(batch_canaries, baseline)
            drifted = drift is not None and abs(drift) > COLLECT.DRIFT_LIMIT_PCT
            drifted_streak = drifted_streak + 1 if drifted else 0
            if drifted_streak >= COLLECT.DRIFT_BATCHES:
                failure = (
                    f"canary median drift {drift:+.2f}% beyond {COLLECT.DRIFT_LIMIT_PCT}% for "
                    f"{drifted_streak} consecutive batches"
                )
            else:
                for sample_id in batch_ids:
                    sketch_id, position = by_id[sample_id]
                    result = measure(position, sketches[sketch_id], layouts[sketch_id], config, executor)
                    batch_rows.append(
                        {
                            **COLLECT._record(sample_id, sketch_id, position, result, batch_id),
                            "retime_order": position_of[sample_id],
                        }
                    )
        except Exception as error:  # an RPC failure ends a one-session run
            failure = f"{type(error).__name__}: {error}"
        after, after_errors = _probe(args.ssh_target)
        device_state = COLLECT._device_state_record(before, after, errors + after_errors, (0, 0))
        for row in batch_rows + batch_canaries:
            row["target_id"] = target_config.target_id
            row["rpc_session_index"] = 0
            row["device_state"] = device_state
        rows.extend(batch_rows)
        canary_rows.extend(batch_canaries)
        frequency = COLLECT.batch_frequency_check(before, after, batch_rows + batch_canaries, config)
        batch_status = Counter(row["status"] for row in batch_rows)
        non_ok = sum(count for name, count in batch_status.items() if name != "ok")
        temperatures = COLLECT._temperatures_c(before, after)
        if failure is None:
            if after is None:
                failure = f"device-state probe failed after batch {batch_id}: {after_errors}"
            else:
                try:
                    check_server(after, rpc_server)
                except RuntimeError as error:
                    failure = str(error)
        if failure is None and frequency["drop"]:
            failure = "frequency drop under load: " + "; ".join(frequency["reasons"])
        if failure is None and batch_rows and non_ok / len(batch_rows) > COLLECT.NON_OK_LIMIT:
            failure = (
                f"{non_ok}/{len(batch_rows)} non-ok rows ({dict(batch_status)}) exceed "
                f"{COLLECT.NON_OK_LIMIT:.0%}"
            )
        record = {
            "batch_id": batch_id,
            "started_utc": batch_started_utc,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "batch_wall_s": time.monotonic() - batch_started,
            "rows": len(batch_rows),
            "status_counts": dict(batch_status),
            "canary_drift_pct": drift,
            "canary_baseline": baseline_source,
            "canary_ok": sum(1 for row in batch_canaries if row["status"] == "ok"),
            "frequency": frequency,
            "temperature_c": {"min": min(temperatures, default=None), "max": max(temperatures, default=None)},
            "rpc_session_index": {"batch_start": 0, "batch_end": 0},
            "stop_reason": failure,
        }
        with outputs["batches"].open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if rows:
            _write_parquet(rows, outputs["partial"])
        if canary_rows:
            _write_parquet(canary_rows, outputs["canary"])
        write_meta()
        drift_text = "n/a" if drift is None else f"{drift:+.2f}%"
        mean_mhz = frequency.get("mean_mhz")
        print(
            f"Batch {batch_id + 1}/{len(batches)}: {len(batch_rows)} rows {dict(batch_status)} | "
            f"total {len(rows)}/{len(order)} | canary drift {drift_text} | CPU mean "
            f"{'n/a' if mean_mhz is None else f'{mean_mhz:.0f}'} MHz, "
            f"{frequency.get('seconds_at_max', 0):.0f} s at max vs {frequency['busy_estimate_s']:.0f} s busy | "
            f"max temp {max(temperatures, default=float('nan')):.1f} C | "
            f"elapsed {COLLECT._hms(time.monotonic() - run_started)}",
            flush=True,
        )
        if failure is not None:
            stop(failure)

    inputs_after = _input_hashes(data_dir)
    if inputs_after != inputs_before:
        stop(f"input files changed during the re-timing: {inputs_before} != {inputs_after}")
    measured_ids = [int(row["id"]) for row in rows]
    if sorted(measured_ids) != sorted(test_ids):
        stop("the re-timed ids do not match the TEST ids")
    _write_parquet(rows, outputs["pool"])
    outputs["partial"].unlink()
    status.update(state="complete")
    write_meta(inputs_after)
    print(
        f"Re-timing finished: {len(rows)}/{len(order)} TEST schedules "
        f"{dict(Counter(row['status'] for row in rows))} in one RPC session -> {outputs['pool']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
