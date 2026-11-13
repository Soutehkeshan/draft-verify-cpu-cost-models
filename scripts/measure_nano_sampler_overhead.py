#!/usr/bin/env python3
"""Measure telemetry sampler CPU cost while the Nano RPC pool is idle."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.rpc import RPCExecutor
from dvcm.telemetry import NanoTelemetry


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(int((len(ordered) - 1) * percentile), len(ordered) - 1)
    return ordered[index]


def _measure_mode(
    ssh_target: str,
    rpc_port: int,
    interval_ms: int,
    hardware_interval_ms: int,
    duration_seconds: float,
    thread_only: bool,
) -> dict[str, Any]:
    sampler = NanoTelemetry(
        ssh_target,
        interval_ms=interval_ms,
        hardware_interval_ms=hardware_interval_ms,
        rpc_port=rpc_port,
        thread_only=thread_only,
    )
    sampler.start()
    try:
        time.sleep(duration_seconds)
    finally:
        report = sampler.close()

    shutdown = report["shutdown"] or {}
    elapsed = shutdown.get("sampler_elapsed_seconds")
    cpu_seconds = shutdown.get("sampler_cpu_seconds")
    sample_cpu_ms = [
        sample["sample_cpu_duration_ms"] for sample in report["samples"]
    ]
    return {
        "mode": "thread-only" if thread_only else "full",
        "thread_only": thread_only,
        "duration_seconds_requested": duration_seconds,
        "sampler_elapsed_seconds": elapsed,
        "sampler_cpu_seconds": cpu_seconds,
        "sampler_one_core_cpu_percent": (
            100.0 * cpu_seconds / elapsed
            if elapsed and cpu_seconds is not None
            else None
        ),
        "sample_count": len(report["samples"]),
        "sample_cpu_duration_ms_mean": (
            mean(sample_cpu_ms) if sample_cpu_ms else None
        ),
        "sample_cpu_duration_ms_p95": _percentile(sample_cpu_ms, 0.95),
        "sample_cpu_duration_ms_max": max(sample_cpu_ms) if sample_cpu_ms else None,
        "server_thread_environment": report["metadata"][
            "rpc_server_thread_environment"
        ],
        "server_processes": {
            key: report["metadata"][key]
            for key in ("rpc_master_pid", "rpc_listener_pid", "rpc_session_pid")
        },
        "cleanup_error": report["cleanup_error"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument("--device-rpc-port", type=int)
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument("--hardware-interval-ms", type=int, default=1000)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    args = parser.parse_args()
    device_rpc_port = (
        args.device_rpc_port
        if args.device_rpc_port is not None
        else args.rpc_port
    )
    for label, port in (
        ("--rpc-port", args.rpc_port),
        ("--device-rpc-port", device_rpc_port),
    ):
        if not 1 <= port <= 65535:
            parser.error("{} must be between 1 and 65535".format(label))
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be positive")

    executor = RPCExecutor.connect(args.rpc_host, args.rpc_port)
    effective_threads = int(
        executor.session.get_function("runtime.NumThreads")()
    )
    results = [
        _measure_mode(
            args.ssh_target,
            device_rpc_port,
            args.interval_ms,
            args.hardware_interval_ms,
            args.duration_seconds,
            thread_only,
        )
        for thread_only in (True, False)
    ]
    print(
        json.dumps(
            {
                "runtime_num_threads": effective_threads,
                "sampling_interval_ms": args.interval_ms,
                "hardware_sampling_interval_ms": args.hardware_interval_ms,
                "results": results,
                "dataset_written": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(not item["cleanup_error"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
