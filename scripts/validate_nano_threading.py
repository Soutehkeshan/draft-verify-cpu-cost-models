#!/usr/bin/env python3
"""Validate Nano RPC thread-pool settings and a parallel CPU workload."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import tvm
from tvm import te

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.config import load_target_config
from dvcm.rpc import RPCExecutor
from dvcm.telemetry import NanoTelemetry


def _parallel_scale(n: int) -> Any:
    x = te.placeholder((n,), dtype="float32", name="x")
    y = te.compute((n,), lambda i: 2.0 * x[i] + 1.0, name="y")
    prim = te.create_prim_func([x, y]).with_attr(
        "global_symbol", "parallel_scale"
    )
    schedule = tvm.s_tir.Schedule(tvm.IRModule.from_expr(prim))
    block = schedule.get_sblock("y", func_name="parallel_scale")
    schedule.parallel(schedule.get_loops(block)[0])
    return schedule.mod


def _timing_milliseconds(
    module: Any,
    device: Any,
    x: Any,
    y: Any,
    repeat: int,
    min_repeat_ms: int,
) -> list[float]:
    result = module.time_evaluator(
        "parallel_scale",
        device,
        repeat=repeat,
        number=1,
        min_repeat_ms=min_repeat_ms,
    )(x, y)
    return [float(value) * 1000.0 for value in result.results]


def _pool_worker_ids(pool_report: dict[str, Any], activity_report: dict[str, Any] | None = None) -> list[int]:
    """Return TIDs of TVM thread-pool workers.

    Prefer pool_report (threads seen during the idle pool phase).  If that yields
    nothing — because the pool was created lazily after pool_monitor closed — fall
    back to threads present at activity_monitor start that are not the session PID.
    """
    metadata = pool_report["metadata"]
    initial = set(metadata["rpc_server_thread_ids_at_start"])
    observed = {
        int(tid)
        for sample in pool_report["samples"]
        for tid in sample["rpc_thread_runtime_ns"]
    }
    workers = sorted(observed - initial - {metadata["rpc_session_pid"]})
    if workers or activity_report is None:
        return workers
    # Fall back: workers existed at the start of the activity monitor
    act_meta = activity_report["metadata"]
    at_start = set(act_meta["rpc_server_thread_ids_at_start"])
    return sorted(at_start - {act_meta["rpc_session_pid"]})


def _activity_summary(
    report: dict[str, Any], measurement_id: str, pool_worker_tids: list[int]
) -> dict[str, Any]:
    samples = [
        sample
        for sample in report["samples"]
        if sample["active_measurement_id"] == measurement_id
    ]
    session_pid = report["metadata"]["rpc_session_pid"]
    pool_worker_set = set(pool_worker_tids)
    active_tids = set()
    active_cpus = set()
    max_active_workers = 0
    for sample in samples:
        sample_active_tids = set(sample["active_rpc_tids"])
        active_workers = sample_active_tids.intersection(pool_worker_set)
        max_active_workers = max(max_active_workers, len(active_workers))
        active_tids.update(active_workers)
        if session_pid in sample_active_tids:
            active_tids.add(session_pid)
        relevant_tids = active_workers | (
            {session_pid} if session_pid in sample_active_tids else set()
        )
        active_cpus.update(
            int(sample["rpc_thread_processor_ids"][str(tid)])
            for tid in relevant_tids
        )
    worker_tids = sorted(tid for tid in active_tids if tid != session_pid)
    active_cpus = sorted(active_cpus)
    return {
        "sample_count": len(samples),
        "max_active_workers": max_active_workers,
        "active_worker_tids": worker_tids,
        "active_cpu_ids": active_cpus,
        "all_four_cpu_ids_observed": active_cpus == [0, 1, 2, 3],
        "sample_cpu_duration_ms_mean": (
            mean(sample["sample_cpu_duration_ms"] for sample in samples)
            if samples
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--device-rpc-port", type=int)
    parser.add_argument("--expected-threads", type=int, choices=(1, 4), required=True)
    parser.add_argument("--elements", type=int, default=1_048_576)
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument("--min-repeat-ms", type=int, default=100)
    parser.add_argument("--active-repeat-ms", type=int, default=5000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    device_rpc_port = (
        args.device_rpc_port if args.device_rpc_port is not None else args.port
    )
    if not 1 <= device_rpc_port <= 65535:
        parser.error("--device-rpc-port must be between 1 and 65535")
    if args.elements < 1:
        parser.error("--elements must be positive")

    target_config = load_target_config("jetson_nano", ROOT)
    target = target_config.make_tvm_target()
    executor = RPCExecutor.connect(args.host, args.port)

    pool_monitor = NanoTelemetry(
        args.ssh_target,
        interval_ms=args.interval_ms,
        thread_only=True,
        rpc_port=device_rpc_port,
    )
    pool_monitor.start()
    try:
        effective_threads = int(
            executor.session.get_function("runtime.NumThreads")()
        )
        time.sleep(max(args.interval_ms * 1.5 / 1000.0, 0.1))
    finally:
        pool_report = pool_monitor.close()

    expected_environment = {
        "TVM_NUM_THREADS": str(args.expected_threads),
        "OMP_NUM_THREADS": str(args.expected_threads),
        "TVM_BIND_THREADS": "1",
        "TVM_EXCLUDE_WORKER0": "1",
        "TVM_THREAD_POOL_SPIN_COUNT": "0",
    }
    actual_environment = pool_report["metadata"][
        "rpc_server_thread_environment"
    ]
    config_errors = {
        key: {"expected": value, "actual": actual_environment.get(key)}
        for key, value in expected_environment.items()
        if actual_environment.get(key) != value
    }
    if effective_threads != args.expected_threads:
        config_errors["runtime.NumThreads"] = {
            "expected": args.expected_threads,
            "actual": effective_threads,
        }

    mod = _parallel_scale(args.elements)
    compile_started = time.perf_counter()
    library = tvm.compile(mod, target=target)
    compile_time_s = time.perf_counter() - compile_started
    remote_library = executor.load_module(library)
    remote_module = remote_library.module

    rng = np.random.default_rng(42)
    x_np = rng.random(args.elements, dtype=np.float32)
    expected = 2.0 * x_np + 1.0
    to_device = getattr(tvm.runtime, "tensor", None) or tvm.nd.array
    input_started = time.perf_counter()
    x = to_device(x_np, executor.device)
    y = to_device(np.zeros_like(x_np), executor.device)
    input_upload_time_s = time.perf_counter() - input_started
    remote_module["parallel_scale"](x, y)
    np.testing.assert_allclose(y.numpy(), expected, rtol=0.0, atol=1e-6)

    clean_timings_ms = _timing_milliseconds(
        remote_module,
        executor.device,
        x,
        y,
        repeat=3,
        min_repeat_ms=args.min_repeat_ms,
    )

    activity_id = "parallel-vector-active-workers"
    activity_monitor = NanoTelemetry(
        args.ssh_target,
        interval_ms=args.interval_ms,
        thread_only=True,
        rpc_port=device_rpc_port,
    )
    activity_monitor.start()
    activity_monitor.mark("start", activity_id)
    try:
        _timing_milliseconds(
            remote_module,
            executor.device,
            x,
            y,
            repeat=1,
            min_repeat_ms=args.active_repeat_ms,
        )
    finally:
        activity_monitor.mark("end", activity_id)
        activity_report = activity_monitor.close()

    pool_workers = _pool_worker_ids(pool_report, activity_report)
    activity = _activity_summary(activity_report, activity_id, pool_workers)
    report = {
        "target": str(target),
        "tvm_version": str(tvm.__version__),
        "elements": args.elements,
        "expected_threads": args.expected_threads,
        "runtime_num_threads": effective_threads,
        "server_thread_environment": actual_environment,
        "server_processes": {
            key: pool_report["metadata"][key]
            for key in ("rpc_master_pid", "rpc_listener_pid", "rpc_session_pid")
        },
        "thread_pool_worker_tids_created": pool_workers,
        "thread_pool_worker_count_created": len(pool_workers),
        "config_errors": config_errors,
        "compile_time_s": compile_time_s,
        "export_time_s": remote_library.export_time_s,
        "upload_time_s": remote_library.upload_time_s,
        "link_load_time_s": remote_library.link_load_time_s,
        "input_upload_time_s": input_upload_time_s,
        "correctness": "passed",
        "clean_timing_sampler_concurrent": False,
        "clean_timings_ms": clean_timings_ms,
        "clean_median_ms": float(np.median(clean_timings_ms)),
        "worker_activity_test": "parallel workload only; timing result omitted",
        "worker_activity": activity,
        "sampler_cpu_seconds_during_worker_test": (
            activity_report["shutdown"]["sampler_cpu_seconds"]
        ),
        "worker_activity_observation_elapsed_seconds": (
            activity_report["shutdown"]["sampler_elapsed_seconds"]
        ),
        "dataset_written": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if config_errors:
        return 1
    if args.expected_threads == 4 and (
        activity["max_active_workers"] < 3
        or not activity["all_four_cpu_ids_observed"]
    ):
        return 1
    if args.expected_threads == 1 and activity["max_active_workers"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
