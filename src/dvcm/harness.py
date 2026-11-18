"""Compile individual MetaSchedule traces and time them on the Nano over RPC."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import numpy as np
import tvm
from tvm import te
from tvm.s_tir import meta_schedule as ms

from dvcm.rpc import RPCExecutor

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarnessConfig:
    target: Any
    postprocessors: str
    n: int
    repeats: int
    number: int
    min_repeat_ms: int
    seed: int
    slow_threshold_ms: float

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive")
        if self.repeats < 1 or self.number < 1:
            raise ValueError("repeats and number must be positive")
        if self.min_repeat_ms < 0:
            raise ValueError("min_repeat_ms must be non-negative")
        if not self.slow_threshold_ms > 0:
            raise ValueError("slow_threshold_ms must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class TimingResult:
    repeats_ms: tuple[float, ...]
    median_ms: float
    std_ms: float
    status: str
    max_abs_error: float | None = None


@dataclass(frozen=True)
class MeasurementResult:
    timing: TimingResult
    build_time_s: float
    export_time_s: float = 0.0
    upload_time_s: float = 0.0
    link_load_time_s: float = 0.0


def make_matmul(n: int) -> tvm.tirx.PrimFunc:
    """Return a TIR PrimFunc computing C = A @ B for n x n float32 matrices."""
    a = te.placeholder((n, n), dtype="float32", name="A")
    b = te.placeholder((n, n), dtype="float32", name="B")
    k = te.reduce_axis((0, n), name="k")
    c = te.compute((n, n), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), name="C")
    return te.create_prim_func([a, b, c])


def _zero_output(c: Any, config: HarnessConfig) -> None:
    c.copyfrom(np.zeros((config.n, config.n), dtype="float32"))


def _stats(
    values: tuple[float, ...], status: str, max_abs_error: float | None = None
) -> TimingResult:
    return TimingResult(
        repeats_ms=values,
        median_ms=float(np.median(values)),
        std_ms=float(np.std(values)),
        status=status,
        max_abs_error=max_abs_error,
    )


def time_module(lib: Any, config: HarnessConfig, rpc_executor: RPCExecutor) -> TimingResult:
    """Check correctness, then time a module loaded on the remote CPU."""
    a, b, c, reference = rpc_executor.prepare_inputs(config.n, config.seed)
    device = rpc_executor.device
    mod = lib
    entry = mod["main"]
    _zero_output(c, config)
    entry(a, b, c)
    output = c.numpy()
    max_abs_error = float(np.max(np.abs(output - reference)))
    try:
        np.testing.assert_allclose(output, reference, rtol=1e-4, atol=1e-4)
    except AssertionError:
        LOGGER.exception("Compiled schedule returned an incorrect result")
        return TimingResult(
            (), float("nan"), float("nan"), "wrong_result", max_abs_error
        )

    single_result = mod.time_evaluator("main", device, repeat=1, number=1, min_repeat_ms=0)(a, b, c)
    single_ms = float(single_result.results[0]) * 1000.0
    if single_ms > config.slow_threshold_ms:
        return _stats((single_ms,), "slow_skip", max_abs_error)

    evaluator = mod.time_evaluator(
        "main",
        device,
        repeat=config.repeats,
        number=config.number,
        min_repeat_ms=config.min_repeat_ms,
    )
    result = evaluator(a, b, c)
    repeats_ms = tuple(float(value) * 1000.0 for value in result.results)
    return _stats(repeats_ms, "ok", max_abs_error)


def _schedule_from_position(
    position: np.ndarray,
    design_space_trace: Any,
    layout: dict[str, Any],
    config: HarnessConfig,
) -> Any:
    from dvcm.encoding import position_to_decisions

    trace = position_to_decisions(position, design_space_trace, layout)
    if not hasattr(design_space_trace, "trace"):
        raise ValueError("design_space_trace must be a MetaSchedule Schedule")
    schedule = tvm.s_tir.Schedule(make_matmul(config.n), seed=max(1, config.seed))
    trace.apply_to_schedule(schedule, remove_postproc=False)
    for postproc in ms.Postproc.create(config.postprocessors):
        if not postproc.apply(schedule):
            raise ValueError(
                f"{config.postprocessors} postprocessor rejected schedule: {postproc}"
            )
    return schedule


def build_from_position(
    position: np.ndarray,
    design_space_trace: Any,
    layout: dict[str, Any],
    config: HarnessConfig,
) -> tuple[Any | None, str]:
    """Replay a position on a fresh schedule and compile it for the configured target."""
    try:
        schedule = _schedule_from_position(position, design_space_trace, layout, config)
        return tvm.compile(schedule.mod, target=config.target), "ok"
    except Exception:
        LOGGER.exception("Failed to build encoded schedule")
        return None, "build_fail"


def measure(
    position: np.ndarray,
    design_space_trace: Any,
    layout: dict[str, Any],
    config: HarnessConfig,
    rpc_executor: RPCExecutor,
) -> MeasurementResult:
    """Build one schedule, load it on the Nano, and time it there."""
    start = time.perf_counter()
    lib, status = build_from_position(position, design_space_trace, layout, config)
    build_time_s = time.perf_counter() - start
    if lib is None:
        return MeasurementResult(
            TimingResult((), float("nan"), float("nan"), status),
            build_time_s,
        )
    remote_library = rpc_executor.load_module(lib)
    timing = time_module(remote_library.module, config, rpc_executor)
    return MeasurementResult(
        timing=timing,
        build_time_s=build_time_s,
        export_time_s=remote_library.export_time_s,
        upload_time_s=remote_library.upload_time_s,
        link_load_time_s=remote_library.link_load_time_s,
    )
