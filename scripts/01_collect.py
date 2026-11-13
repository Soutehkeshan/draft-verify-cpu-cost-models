
#!/usr/bin/env python3
"""Collect a reproducible pool of measured CPU schedules."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import tvm
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.encoding import (
    describe_space,
    layout_to_json,
    random_position,
    repair,
    structural_layout,
)
from dvcm.config import (
    TargetConfig,
    WorkloadConfig,
    add_target_argument,
    load_target_config,
    load_workload_config,
)
from dvcm.device_state import (
    frequency_residency,
    read_device_state,
    summarize_device_state,
)
from dvcm.harness import (
    HarnessConfig,
    MeasurementResult,
    TimingResult,
    make_matmul,
    measure,
)
from dvcm.provenance import capture_provenance
from dvcm.rpc import RPCExecutor
from dvcm.study import add_schedule_if_new, save_json

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
    measurement: MeasurementResult,
    batch_id: int,
) -> dict:
    result = measurement.timing
    return {
        "id": int(sample_id),
        "sketch_id": int(sketch_id),
        "vector": [int(value) for value in position],
        "repeats_ms": list(result.repeats_ms),
        "median_ms": result.median_ms,
        "std_ms": result.std_ms,
        "status": result.status,
        "build_time_s": float(measurement.build_time_s),
        "export_time_s": float(measurement.export_time_s),
        "upload_time_s": float(measurement.upload_time_s),
        "link_load_time_s": float(measurement.link_load_time_s),
        "batch_id": int(batch_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _execution_metadata(
    target_config: TargetConfig,
    workload: WorkloadConfig,
    target: tvm.target.Target,
    rpc_executor: RPCExecutor | None,
    rpc_host: str | None,
    rpc_port: int,
) -> dict[str, Any]:
    return {
        "target_id": target_config.target_id,
        "target_profile": target_config.to_dict(),
        "workload": workload.to_dict(),
        "tvm_version": tvm.__version__,
        "target": str(target),
        "execution_backend": "rpc" if rpc_executor is not None else "local",
        "rpc_host": rpc_host,
        "rpc_port": rpc_port if rpc_executor is not None else None,
        "rpc_input_prepare_time_s": (
            rpc_executor.input_prepare_time_s if rpc_executor is not None else 0.0
        ),
        "rpc_input_upload_time_s": (
            rpc_executor.input_upload_time_s if rpc_executor is not None else 0.0
        ),
    }


# --------------------------------------------------------------------------
# Positions-driven collection (e.g. the M2 pool's schedules on the Nano).
# Written for the Nano; not run until the jetson_nano collection gate opens.


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def order_test_first(splits: dict[str, Any], available_ids: set[int]) -> list[int]:
    """TEST ids (stored order), then the nested TRAIN subsets in their existing order.

    Nested subsets are prefixes of one shuffled TRAIN order, so after TEST the run
    grows the largest complete nested subset. VALIDATION is not part of the
    requested order and is appended last.
    """
    order: list[int] = []
    seen: set[int] = set()

    def extend(values: Any) -> None:
        for value in values:
            value = int(value)
            if value not in seen:
                seen.add(value)
                order.append(value)

    extend(splits["test"])
    nested = splits["nested_train"]
    for name in sorted((key for key in nested if key != "all"), key=int):
        extend(nested[name])
    extend(nested.get("all", ()))
    extend(splits["train"])
    extend(splits["validation"])
    unknown = sorted(seen - available_ids)
    if unknown:
        raise ValueError(f"splits reference ids missing from the source pool: {unknown[:5]}")
    missing = sorted(available_ids - seen)
    if missing:
        raise ValueError(f"source pool ids missing from the splits: {missing[:5]}")
    return order


def require_pilot_threshold(output_dir: Path) -> dict[str, Any]:
    """Refuse to collect without the pilot's slow threshold and its pilot report."""
    meta_path = output_dir / "run_meta.json"
    report_path = output_dir / "pilot_report.json"
    if not meta_path.is_file():
        raise SystemExit(f"Refusing to collect: {meta_path} does not exist; run the pilot first")
    with meta_path.open(encoding="utf-8") as source:
        meta = json.load(source)
    threshold = meta.get("slow_threshold_ms")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or threshold <= 0
    ):
        raise SystemExit(f"Refusing to collect: no positive slow_threshold_ms in {meta_path}")
    if not report_path.is_file():
        raise SystemExit(f"Refusing to collect: pilot report {report_path} does not exist")
    with report_path.open(encoding="utf-8") as source:
        report = json.load(source)
    if report.get("slow_threshold_ms") != threshold:
        raise SystemExit(
            "Refusing to collect: run_meta.json slow_threshold_ms does not match the pilot report"
        )
    recorded = (meta.get("pilot_report") or {}).get("sha256")
    if recorded != _sha256(report_path):
        raise SystemExit(
            "Refusing to collect: the pilot report checksum in run_meta.json is missing or stale"
        )
    if not (meta.get("rpc_server") or {}).get("environment"):
        raise SystemExit("Refusing to collect: run_meta.json does not record the RPC server environment")
    return meta


class RemoteSession:
    """Owns the RPC executor so a dropped connection can be replaced."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.executor = RPCExecutor.connect(host, port)
        self.reconnects = 0

    def reconnect(self) -> None:
        self.executor = RPCExecutor.connect(self.host, self.port)
        self.reconnects += 1


def measure_with_retry(
    session: RemoteSession,
    position: np.ndarray,
    sketch: Any,
    layout: dict,
    config: HarnessConfig,
    log_error: Callable[[int, BaseException], None],
) -> MeasurementResult:
    """Measure once; on an RPC failure reconnect and retry once, else status rpc_error.

    Build failures are handled inside measure(). After a failed retry the session is
    reconnected again so the next schedule starts on a live connection; if any
    reconnect fails the exception stops the run (the caller checkpoints first).
    """
    for attempt in (1, 2):
        try:
            return measure(position, sketch, layout, config, session.executor)
        except Exception as error:  # remote failure after a successful build
            log_error(attempt, error)
            session.reconnect()
    return MeasurementResult(
        TimingResult((), float("nan"), float("nan"), "rpc_error"), float("nan")
    )


def _verify_layouts(layouts: list[dict], source_dir: Path, output_dir: Path) -> None:
    source_files = sorted(source_dir.glob("layout_*.json"))
    if len(source_files) != len(layouts):
        raise SystemExit(
            f"Refusing to collect: {len(source_files)} source layouts but {len(layouts)} sketches"
        )
    for sketch_id, layout in enumerate(layouts):
        current = layout_to_json(layout, sketch_id)
        with (source_dir / f"layout_{sketch_id}.json").open(encoding="utf-8") as source:
            if structural_layout(json.load(source)) != current:
                raise SystemExit(
                    f"Refusing to collect: sketch {sketch_id} design space differs from the source layout"
                )
        out_path = output_dir / f"layout_{sketch_id}.json"
        if out_path.exists():
            with out_path.open(encoding="utf-8") as source:
                if json.load(source) != current:
                    raise SystemExit(f"Refusing to collect: {out_path} differs from the design space")
        else:
            save_json(out_path, current)


def _require_valid(position: np.ndarray, layout: dict, label: str) -> None:
    if not np.array_equal(repair(position.astype(float), layout), position.astype(float)):
        raise SystemExit(f"Refusing to collect: {label} is not a valid position (no resampling)")


def _probe(ssh_target: str) -> tuple[dict | None, list[str]]:
    try:
        return read_device_state(ssh_target), []
    except Exception as error:
        return None, [f"{type(error).__name__}: {error}"]


def _check_server_environment(state: dict, expected: dict[str, str]) -> None:
    wanted = {k: v for k, v in expected.items() if k.startswith(("TVM_", "OMP_"))}
    servers = state.get("rpc_servers", [])
    if not any(server.get("environment") == wanted for server in servers):
        raise RuntimeError(
            f"RPC server environment differs from run_meta.json ({wanted}): {servers}"
        )


def _device_state_record(
    before: dict | None, after: dict | None, errors: list[str], rpc_sessions: tuple[int, int]
) -> str:
    # Schedules can shift by several percent between RPC sessions (qualification
    # noise test), so record which session(s) measured the batch.
    return json.dumps(
        {
            "before": summarize_device_state(before) if before else None,
            "after": summarize_device_state(after) if after else None,
            "frequency_residency": (
                frequency_residency(before, after) if before and after else None
            ),
            "rpc_session_index": {"batch_start": rpc_sessions[0], "batch_end": rpc_sessions[1]},
            "errors": errors,
        },
        sort_keys=True,
    )


def _append_log(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    incoming = pd.DataFrame(rows)
    if path.exists():
        incoming = pd.concat([pd.read_parquet(path), incoming], ignore_index=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    incoming.to_parquet(temporary, index=False)
    temporary.replace(path)


def _log_error(path: Path, label: Any, attempt: int, error: BaseException) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schedule": label,
        "attempt": attempt,
        "error": f"{type(error).__name__}: {error}",
    }
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record) + "\n")
    print(f"RPC failure on {label} (attempt {attempt}): {record['error']}", file=sys.stderr)


def collect_from_positions(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    target_config: TargetConfig,
    workload: WorkloadConfig,
) -> None:
    """Measure an existing pool's (sketch_id, vector) rows on an RPC target."""
    if args.smoke or args.pilot_only:
        parser.error("--positions-from cannot be combined with --smoke or --pilot-only")
    if args.order != "test_first":
        parser.error("--positions-from requires --order test_first")
    if not args.rpc_host:
        parser.error("--positions-from requires --rpc-host")
    if not args.ssh_target:
        parser.error("--positions-from requires --ssh-target for device-state records")
    if args.n < 1:
        parser.error("--n must be positive")
    source = Path(args.positions_from).resolve()
    splits_path = Path(args.splits or source.parent / "splits.json").resolve()
    data_dir, _ = target_config.paths(ROOT)
    output_dir = Path(args.output_dir or data_dir).resolve()
    if output_dir == source.parent:
        parser.error("the output directory must differ from the source pool's directory")

    # Refusals come before any device contact or dataset write.
    run_meta = require_pilot_threshold(output_dir)
    target_config.require_benchmarking_profile()
    provenance = capture_provenance(ROOT, allow_dirty=bool(getattr(args, "allow_dirty", False)))
    canary_path = output_dir / "canary.json"
    if not canary_path.is_file():
        raise SystemExit(f"Refusing to collect: qualified canaries {canary_path} do not exist")
    with canary_path.open(encoding="utf-8") as stream:
        canaries = json.load(stream)
    threshold = float(run_meta["slow_threshold_ms"])
    target = target_config.make_tvm_target()

    positions = pd.read_parquet(source, columns=["id", "sketch_id", "vector"])
    with splits_path.open(encoding="utf-8") as stream:
        splits = json.load(stream)
    order = order_test_first(splits, set(positions["id"].astype(int)))[: args.n]
    by_id = {
        int(row.id): (int(row.sketch_id), np.asarray(row.vector, dtype=int))
        for row in positions.itertuples()
    }
    context = ms.TuneContext(
        mod=make_matmul(workload.n),
        target=target,
        space_generator=PostOrderApply(
            sch_rules=target_config.schedule_rules,
            postprocs=target_config.postprocessors,
        ),
        rand_state=workload.seed,
        num_threads=target_config.thread_settings.get("TVM_NUM_THREADS", 1),
    )
    sketches = context.generate_design_space()
    layouts = [describe_space(schedule.trace) for schedule in sketches]
    _verify_layouts(layouts, source.parent, output_dir)
    for sample_id in order:
        sketch_id, position = by_id[sample_id]
        _require_valid(position, layouts[sketch_id], f"id {sample_id}")
    for index, item in enumerate(canaries):
        _require_valid(
            np.asarray(item["vector"], dtype=int), layouts[int(item["sketch_id"])], f"canary {index}"
        )

    pool_path = output_dir / "pool.parquet"
    canary_log_path = output_dir / "canary_log.parquet"
    error_log_path = output_dir / "collection_errors.jsonl"
    meta_path = output_dir / "run_meta.json"
    existing = pd.read_parquet(pool_path) if pool_path.exists() else pd.DataFrame()
    measured = set(existing["id"].astype(int)) if not existing.empty else set()
    config = HarnessConfig(
        target=target,
        postprocessors=target_config.postprocessors,
        n=workload.n,
        repeats=workload.measurement["repeats"],
        number=workload.measurement["number"],
        min_repeat_ms=workload.measurement["min_repeat_ms"],
        seed=workload.seed,
        slow_threshold_ms=threshold,
    )
    expected_environment = run_meta["rpc_server"]["environment"]
    first_start = (run_meta.get("collection") or {}).get("start_timestamp")
    start_timestamp = first_start or datetime.now(timezone.utc).isoformat()
    session = RemoteSession(args.rpc_host, args.rpc_port)

    def save_progress() -> None:
        pool = pd.read_parquet(pool_path) if pool_path.exists() else pd.DataFrame()
        statuses = Counter(pool["status"]) if not pool.empty else Counter()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["collection"] = {
            **_execution_metadata(
                target_config, workload, target, session.executor, args.rpc_host, args.rpc_port
            ),
            "source_pool": str(source),
            "source_pool_sha256": _sha256(source),
            "splits": str(splits_path),
            "splits_sha256": _sha256(splits_path),
            "order": "test_first",
            "order_detail": "TEST (stored order), nested TRAIN subsets (existing order), then VALIDATION",
            "n_target": len(order),
            "batch_size": BATCH_SIZE,
            "cooldown_s": COOLDOWN_S,
            "canaries": len(canaries),
            "slow_threshold_ms": threshold,
            "repeats": config.repeats,
            "number": config.number,
            "min_repeat_ms": config.min_repeat_ms,
            "measured": len(pool),
            "status_counts": {name: int(count) for name, count in statuses.items()},
            "collection_complete": set(order) <= measured,
            "reconnects": session.reconnects,
            "start_timestamp": start_timestamp,
            "last_update": datetime.now(timezone.utc).isoformat(),
            "host_platform": platform.platform(),
            "python_version": platform.python_version(),
            "provenance": provenance,
        }
        save_json(meta_path, meta)

    ran_batch = False
    for batch_id, start in enumerate(range(0, len(order), BATCH_SIZE)):
        todo = [i for i in order[start : start + BATCH_SIZE] if i not in measured]
        if not todo:
            continue
        if ran_batch and COOLDOWN_S:
            time.sleep(COOLDOWN_S)
        before, errors = _probe(args.ssh_target)
        if before is None:
            raise RuntimeError(f"Device-state probe failed before batch {batch_id}: {errors}")
        _check_server_environment(before, expected_environment)
        session_at_start = session.reconnects
        rows: list[dict] = []
        canary_rows: list[dict] = []
        try:
            for canary_id, item in enumerate(canaries):
                sketch_id = int(item["sketch_id"])
                position = np.asarray(item["vector"], dtype=int)
                result = measure_with_retry(
                    session, position, sketches[sketch_id], layouts[sketch_id], config,
                    partial(_log_error, error_log_path, f"canary-{canary_id}"),
                )
                canary_rows.append(
                    {
                        **_record(-1, sketch_id, position, result, batch_id),
                        "canary_id": canary_id,
                        "source_id": int(item.get("id", -1)),
                    }
                )
            for sample_id in todo:
                sketch_id, position = by_id[sample_id]
                result = measure_with_retry(
                    session, position, sketches[sketch_id], layouts[sketch_id], config,
                    partial(_log_error, error_log_path, sample_id),
                )
                rows.append(_record(sample_id, sketch_id, position, result, batch_id))
        finally:
            after, after_errors = _probe(args.ssh_target)
            state = _device_state_record(
                before, after, errors + after_errors, (session_at_start, session.reconnects)
            )
            for row in rows + canary_rows:
                row["target_id"] = target_config.target_id
                row["device_state"] = state
            _append_rows(rows, pool_path)  # checkpoint per batch
            _append_log(canary_rows, canary_log_path)
            measured.update(int(row["id"]) for row in rows)
            save_progress()
        ran_batch = True
        done = len(set(order) & measured)
        print(f"Batch {batch_id}: {len(rows)} schedules; {done}/{len(order)} measured", flush=True)
    if not ran_batch:
        print(f"Collection already complete: {len(set(order) & measured)}/{len(order)} measured")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_target_argument(parser)
    parser.add_argument("--n", type=int, default=N_TARGET)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rpc-host")
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument(
        "--positions-from",
        type=Path,
        help="measure the (id, sketch_id, vector) rows of an existing pool, e.g. data/pool.parquet",
    )
    parser.add_argument(
        "--order",
        choices=("test_first",),
        help="with --positions-from: TEST ids, then nested TRAIN subsets, then VALIDATION",
    )
    parser.add_argument("--splits", type=Path, help="splits.json (default: next to --positions-from)")
    parser.add_argument("--ssh-target", help="Nano SSH target for per-batch device-state records")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="with --positions-from: allow an uncommitted worktree (recorded in provenance)",
    )
    args = parser.parse_args()
    target_config = load_target_config(args.target, ROOT)
    workload = load_workload_config(root=ROOT)
    if args.positions_from is not None:
        collect_from_positions(args, parser, target_config, workload)
        return
    if args.order or args.splits or args.ssh_target or args.allow_dirty:
        parser.error("--order, --splits, --ssh-target, and --allow-dirty require --positions-from")
    target_config.require_benchmarking_profile()
    target = target_config.make_tvm_target()
    rpc_executor = (
        RPCExecutor.connect(args.rpc_host, args.rpc_port)
        if args.rpc_host
        else None
    )
    n_target = 30 if args.smoke else args.n
    repeats = 2 if args.smoke else workload.measurement["repeats"]
    number = workload.measurement["number"]
    min_repeat_ms = workload.measurement["min_repeat_ms"]
    cooldown = 0 if args.smoke else COOLDOWN_S
    if args.pilot_only and args.smoke:
        parser.error("--pilot-only cannot be combined with --smoke")
    if n_target < 1:
        parser.error("--n must be positive")

    data_dir, results_dir = target_config.paths(ROOT)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else data_dir / "smoke"
        if args.smoke
        else data_dir
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
    results_dir.mkdir(parents=True, exist_ok=True)
    num_threads = target_config.thread_settings.get("TVM_NUM_THREADS", 1)
    context = ms.TuneContext(
        mod=make_matmul(workload.n),
        target=target,
        space_generator=PostOrderApply(
            sch_rules=target_config.schedule_rules,
            postprocs=target_config.postprocessors,
        ),
        rand_state=workload.seed,
        num_threads=num_threads,
    )
    print(f"Target profile: {target_config.target_id} ({target})")
    sketches = context.generate_design_space()
    layouts = [describe_space(schedule.trace) for schedule in sketches]
    print(f"Design-space sketches: {len(sketches)}")
    for sketch_id, layout in enumerate(layouts):
        layout_path = output_dir / f"layout_{sketch_id}.json"
        if not layout_path.exists():
            save_json(layout_path, layout_to_json(layout, sketch_id))

    rng = np.random.default_rng(workload.seed)
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
    config = HarnessConfig(
        target=target,
        postprocessors=target_config.postprocessors,
        n=workload.n,
        repeats=repeats,
        number=number,
        min_repeat_ms=min_repeat_ms,
        seed=workload.seed,
        slow_threshold_ms=float("inf"),
    )

    pilot_count = min(PILOT_N, len(remaining))
    pending_rows = []
    pilot_sample_ids = []
    pilot_ms = []
    pilot_setup = []
    for sample_id, (sketch_id, position) in remaining[:pilot_count]:
        measurement = measure(
            position, sketches[sketch_id], layouts[sketch_id], config, rpc_executor
        )
        pending_rows.append(_record(sample_id, sketch_id, position, measurement, -1))
        pilot_sample_ids.append(sample_id)
        if len(pending_rows) >= 10:
            _append_rows(pending_rows, pool_path)
            measured_ids.update(int(row["id"]) for row in pending_rows)
            pending_rows.clear()
        if measurement.timing.repeats_ms:
            pilot_ms.append(measurement.timing.median_ms)
        pilot_setup.append(
            measurement.build_time_s
            + measurement.export_time_s
            + measurement.upload_time_s
            + measurement.link_load_time_s
        )
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
        target=target,
        postprocessors=target_config.postprocessors,
        n=workload.n,
        repeats=repeats,
        number=number,
        min_repeat_ms=min_repeat_ms,
        seed=workload.seed,
        slow_threshold_ms=slow_threshold_ms,
    )
    projection_setup = float(np.mean(pilot_setup[:20])) if pilot_setup else 0.0
    projection_measure = float(np.mean(pilot_ms[:20])) / 1000.0 if pilot_ms else 0.0
    batch_count = (n_target + BATCH_SIZE - 1) // BATCH_SIZE
    projected_seconds = (
        (n_target + batch_count * len(canaries))
        * (projection_setup + projection_measure)
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
            * (projection_setup + projection_measure)
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
            measurement = measure(
                position, sketches[sketch_id], layouts[sketch_id], config, rpc_executor
            )
            batch_canary_rows.append(
                {
                    **_record(-1, sketch_id, position, measurement, batch_id),
                    "canary_id": canary_id,
                }
            )
        canary_rows.extend(batch_canary_rows)
        last_active_batch = batch_id

        for sample_id in batch_indexes:
            sketch_id, position = samples[sample_id]
            measurement = measure(
                position,
                sketches[sketch_id],
                layouts[sketch_id],
                config,
                rpc_executor,
            )
            row = _record(sample_id, sketch_id, position, measurement, batch_id)
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
            else results_dir / "canary.csv"
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
        **_execution_metadata(
            target_config,
            workload,
            target,
            rpc_executor,
            args.rpc_host,
            args.rpc_port,
        ),
        "machine_model": _mac_value(["sysctl", "-n", "hw.model"]),
        "macos_version": platform.mac_ver()[0] or None,
        "host_platform": platform.platform(),
        "python_version": platform.python_version(),
        "power_state": _power_state(),
        "thread_settings": {
            "TVM_NUM_THREADS": str(num_threads),
            "target_num_cores": target_config.num_cores,
        },
        "config": {
            "n": workload.n,
            "seed": workload.seed,
            "repeats": repeats,
            "number": number,
            "min_repeat_ms": min_repeat_ms,
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
