#!/usr/bin/env python3
"""Repeat the three default Nano schedules while sampling live CPU telemetry."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import tvm
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.config import load_target_config, load_workload_config
from dvcm.encoding import decisions_to_position, describe_space
from dvcm.harness import HarnessConfig, make_matmul, measure
from dvcm.rpc import RPCExecutor
from dvcm.telemetry import NanoTelemetry

REPEATS = 7
TIMING_BATCHES = 3
MIN_REPEAT_MS = 20


def _summarize_window(
    telemetry_report: dict[str, Any], measurement_id: str, expected_cores: int
) -> dict[str, Any]:
    markers = [
        marker
        for marker in telemetry_report["markers"]
        if marker["measurement_id"] == measurement_id
    ]
    start = next((marker for marker in markers if marker["action"] == "start"), None)
    end = next((marker for marker in markers if marker["action"] == "end"), None)
    if start is None or end is None:
        return {"sample_count": 0, "marker_error": "start/end marker missing"}

    samples = [
        sample
        for sample in telemetry_report["samples"]
        if sample["active_measurement_id"] == measurement_id
        and start["monotonic_ns"] <= sample["monotonic_ns"] <= end["monotonic_ns"]
    ]
    sample_intervals = [
        sample["interval_since_last_sample_ms"]
        for sample in samples
        if sample.get("interval_since_last_sample_ms") is not None
    ]
    per_core = {}
    active_rpc_cpus = set()
    system_busy_cores = set()
    max_active_non_session_threads = 0
    max_active_threads = 0
    maximum_temperature = {}
    for core in range(expected_cores):
        key = str(core)
        utilizations = [
            sample["cpu_utilization_pct"][key]
            for sample in samples
            if key in sample["cpu_utilization_pct"]
            and sample["cpu_utilization_pct"][key] is not None
        ]
        frequencies = [
            sample["frequency_khz"][key]
            for sample in samples
            if key in sample["frequency_khz"]
            and sample["frequency_khz"][key] is not None
        ]
        governors = sorted(
            {
                sample["governor"][key]
                for sample in samples
                if key in sample["governor"] and sample["governor"][key] is not None
            }
        )
        frequency_sources = sorted(
            {
                sample["frequency_source"][key]
                for sample in samples
                if key in sample["frequency_source"]
                and sample["frequency_source"][key] is not None
            }
        )
        per_core[key] = {
            "mean_system_utilization_pct": mean(utilizations) if utilizations else None,
            "peak_system_utilization_pct": max(utilizations) if utilizations else None,
            "mean_frequency_mhz": (
                mean(frequencies) / 1000.0 if frequencies else None
            ),
            "min_frequency_mhz": (
                min(frequencies) / 1000.0 if frequencies else None
            ),
            "max_frequency_mhz": (
                max(frequencies) / 1000.0 if frequencies else None
            ),
            "frequency_sources": frequency_sources,
            "governors": governors,
        }
        if utilizations and max(utilizations) >= 50.0:
            system_busy_cores.add(core)

    for sample in samples:
        active_rpc_cpus.update(sample["active_rpc_cpu_ids"])
        max_active_non_session_threads = max(
            max_active_non_session_threads,
            sample["active_rpc_non_session_thread_count"],
        )
        max_active_threads = max(
            max_active_threads, sample["active_rpc_thread_count"]
        )
        for zone, temperature in sample["temperatures_c"].items():
            maximum_temperature[zone] = max(
                temperature, maximum_temperature.get(zone, float("-inf"))
            )

    return {
        "sample_count": len(samples),
        "mean_sample_interval_ms": (
            mean(sample_intervals) if sample_intervals else None
        ),
        "max_sample_interval_ms": (
            max(sample_intervals) if sample_intervals else None
        ),
        "window_duration_ms": (
            end["monotonic_ns"] - start["monotonic_ns"]
        )
        / 1_000_000.0,
        "per_core": per_core,
        "max_temperature_c": maximum_temperature,
        "max_active_rpc_non_session_threads": max_active_non_session_threads,
        "max_active_rpc_threads_including_server": max_active_threads,
        "rpc_cpu_ids_observed": sorted(active_rpc_cpus),
        "rpc_active_core_count": len(active_rpc_cpus),
        "system_cores_peak_utilization_at_least_50pct": sorted(system_busy_cores),
        "all_four_rpc_cpu_ids_observed": (
            len(active_rpc_cpus) == expected_cores
        ),
        "all_four_system_cores_peak_at_least_50pct": (
            len(system_busy_cores) == expected_cores
        ),
    }


def _measurement_row(
    sketch_id: int,
    batch_id: int,
    measurement_id: str,
    position: np.ndarray,
    measurement: Any,
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    timing = measurement.timing
    median_ms = timing.median_ms if math.isfinite(timing.median_ms) else None
    std_ms = timing.std_ms if math.isfinite(timing.std_ms) else None
    return {
        "measurement_id": measurement_id,
        "sketch_id": sketch_id,
        "batch_id": batch_id,
        "position": [int(value) for value in position],
        "status": timing.status,
        "repeats_ms": list(timing.repeats_ms),
        "median_ms": median_ms,
        "std_ms": std_ms,
        "build_time_s": measurement.build_time_s,
        "export_time_s": measurement.export_time_s,
        "upload_time_s": measurement.upload_time_s,
        "link_load_time_s": measurement.link_load_time_s,
        "telemetry": telemetry,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument("--device-rpc-port", type=int)
    parser.add_argument("--rpc-pid", type=int)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--hardware-interval-ms", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be between 1 and 65535")
    device_rpc_port = (
        args.device_rpc_port
        if args.device_rpc_port is not None
        else args.rpc_port
    )
    if not 1 <= device_rpc_port <= 65535:
        parser.error("--device-rpc-port must be between 1 and 65535")

    target_config = load_target_config("jetson_nano", ROOT)
    workload = load_workload_config(root=ROOT)
    if target_config.num_cores is None:
        raise ValueError("Nano profile must define num_cores for telemetry")
    target = target_config.make_tvm_target()
    context = ms.TuneContext(
        mod=make_matmul(workload.n),
        target=target,
        space_generator=PostOrderApply(sch_rules="llvm", postprocs="llvm"),
        rand_state=workload.seed,
        num_threads=target_config.thread_settings.get("TVM_NUM_THREADS", 1),
    )
    sketches = context.generate_design_space()
    if len(sketches) != 3:
        raise RuntimeError(
            "Expected the three qualified MetaSchedule sketches; found {}".format(
                len(sketches)
            )
        )
    layouts = [describe_space(sketch.trace) for sketch in sketches]
    positions = [
        decisions_to_position(sketch.trace, layout).astype(int)
        for sketch, layout in zip(sketches, layouts)
    ]
    executor = RPCExecutor.connect(args.rpc_host, args.rpc_port)
    config = HarnessConfig(
        target=target,
        postprocessors="llvm",
        n=workload.n,
        repeats=REPEATS,
        number=1,
        min_repeat_ms=MIN_REPEAT_MS,
        seed=workload.seed,
        slow_threshold_ms=float("inf"),
    )

    telemetry_client = NanoTelemetry(
        args.ssh_target,
        interval_ms=args.interval_ms,
        hardware_interval_ms=args.hardware_interval_ms,
        rpc_pid=args.rpc_pid,
        rpc_port=device_rpc_port,
    )
    telemetry_client.start()
    measurements = []
    try:
        for sketch_id, (sketch, layout, position) in enumerate(
            zip(sketches, layouts, positions)
        ):
            for batch_id in range(TIMING_BATCHES):
                measurement_id = "sketch-{}-batch-{}".format(sketch_id, batch_id)
                result = measure(
                    position,
                    sketch,
                    layout,
                    config,
                    rpc_executor=executor,
                    timing_marker=telemetry_client.mark,
                    timing_label=measurement_id,
                )
                measurements.append((sketch_id, batch_id, measurement_id, position, result))
    finally:
        telemetry_client.close()

    telemetry_report = telemetry_client.report()
    rows = [
        _measurement_row(
            sketch_id,
            batch_id,
            measurement_id,
            position,
            result,
            _summarize_window(
                telemetry_report, measurement_id, target_config.num_cores
            ),
        )
        for sketch_id, batch_id, measurement_id, position, result in measurements
    ]
    report = {
        "target_id": target_config.target_id,
        "target_profile": target_config.to_dict(),
        "target": str(target),
        "target_num_cores": target_config.num_cores,
        "configured_thread_settings": target_config.thread_settings,
        "execution_backend": "rpc",
        "rpc_host": args.rpc_host,
        "rpc_port": args.rpc_port,
        "device_rpc_port": device_rpc_port,
        "tvm_version": str(tvm.__version__),
        "workload": workload.to_dict(),
        "schedule_rules": "llvm",
        "postprocessors": "llvm",
        "sampling_interval_ms": args.interval_ms,
        "hardware_sampling_interval_ms": args.hardware_interval_ms,
        "timing_batches": TIMING_BATCHES,
        "repeats_per_batch": REPEATS,
        "number": 1,
        "min_repeat_ms": MIN_REPEAT_MS,
        "device_metadata": telemetry_report["metadata"],
        "device_shutdown": telemetry_report["shutdown"],
        "telemetry_cleanup_error": telemetry_report["cleanup_error"],
        "measurements": rows,
        "telemetry_samples": telemetry_report["samples"],
        "telemetry_markers": telemetry_report["markers"],
        "telemetry_stderr": telemetry_report["stderr"],
        "dataset_written": False,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(rendered + "\n")
        print("Wrote telemetry report to {}".format(args.output))
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
