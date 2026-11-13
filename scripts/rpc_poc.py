#!/usr/bin/env python3
"""Verify one cross-compiled vector function and one Nano matmul schedule."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tvm
from tvm import te
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.config import load_target_config, load_workload_config
from dvcm.encoding import decisions_to_position, describe_space
from dvcm.harness import HarnessConfig, make_matmul, measure
from dvcm.rpc import RPCExecutor


def _run_vector(target: tvm.target.Target, executor: RPCExecutor) -> None:
    x = te.placeholder((16,), dtype="float32", name="x")
    y = te.compute((16,), lambda i: 2.0 * x[i] + 1.0, name="y")
    prim = te.create_prim_func([x, y]).with_attr(
        "global_symbol", "vector_2x_plus_1"
    )
    start = time.perf_counter()
    library = tvm.compile(tvm.IRModule.from_expr(prim), target=target)
    compile_time_s = time.perf_counter() - start
    remote_library = executor.load_module(library)

    x_np = np.arange(16, dtype=np.float32)
    x_remote = tvm.runtime.tensor(x_np, executor.device)
    y_remote = tvm.runtime.tensor(np.zeros_like(x_np), executor.device)
    remote_library.module["vector_2x_plus_1"](x_remote, y_remote)
    actual = y_remote.numpy()
    np.testing.assert_array_equal(actual, 2.0 * x_np + 1.0)
    result = remote_library.module.time_evaluator(
        "vector_2x_plus_1",
        executor.device,
        number=1,
        repeat=1,
        min_repeat_ms=0,
    )(x_remote, y_remote)
    print(
        "Vector: correct; compile={:.4f}s export={:.4f}s upload={:.4f}s "
        "link/load={:.4f}s remote_kernel_ms={:.6f}".format(
            compile_time_s,
            remote_library.export_time_s,
            remote_library.upload_time_s,
            remote_library.link_load_time_s,
            float(result.results[0]) * 1000.0,
        )
    )


def _run_one_matmul(
    target: tvm.target.Target, executor: RPCExecutor
) -> None:
    target_config = load_target_config("jetson_nano", ROOT)
    workload = load_workload_config(root=ROOT)
    postprocessors = "llvm"
    context = ms.TuneContext(
        mod=make_matmul(workload.n),
        target=target,
        space_generator=PostOrderApply(
            sch_rules="llvm",
            postprocs=postprocessors,
        ),
        rand_state=workload.seed,
        num_threads=target_config.thread_settings.get("TVM_NUM_THREADS", 1),
    )
    sketches = context.generate_design_space()
    if not sketches:
        raise RuntimeError("MetaSchedule produced no candidate sketches")
    sketch = sketches[0]
    layout = describe_space(sketch.trace)
    position = decisions_to_position(sketch.trace, layout).astype(int)
    config = HarnessConfig(
        target=target,
        postprocessors=postprocessors,
        n=workload.n,
        repeats=1,
        number=1,
        min_repeat_ms=0,
        seed=workload.seed,
        slow_threshold_ms=float("inf"),
    )
    measurement = measure(
        position,
        sketch,
        layout,
        config,
        rpc_executor=executor,
    )
    if measurement.timing.status != "ok":
        raise RuntimeError(
            f"First Nano candidate failed: {measurement.timing.status}"
        )
    print(
        "Matmul: one sketch of {}; correct; build={:.4f}s export={:.4f}s "
        "upload={:.4f}s link/load={:.4f}s input_prepare={:.4f}s "
        "input_upload={:.4f}s remote_kernel_ms={:.6f}".format(
            len(sketches),
            measurement.build_time_s,
            measurement.export_time_s,
            measurement.upload_time_s,
            measurement.link_load_time_s,
            executor.input_prepare_time_s,
            executor.input_upload_time_s,
            measurement.timing.median_ms,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    target_config = load_target_config("jetson_nano", ROOT)
    target = target_config.make_tvm_target()
    executor = RPCExecutor.connect(args.host, args.port)
    _run_vector(target, executor)
    _run_one_matmul(target, executor)


if __name__ == "__main__":
    main()
