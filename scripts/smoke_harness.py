#!/usr/bin/env python3
"""Compile and measure ten seeded schedules as a fast integration check."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.encoding import describe_space, random_position
from dvcm.config import LEGACY_TARGET, load_target_config, load_workload_config
from dvcm.harness import HarnessConfig, make_matmul, measure


def main() -> None:
    target_config = load_target_config(LEGACY_TARGET, ROOT)
    workload = load_workload_config(root=ROOT)
    target = target_config.make_tvm_target()
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
    schedule = context.generate_design_space()[0]
    layout = describe_space(schedule.trace)
    config = HarnessConfig(
        target=target,
        postprocessors=target_config.postprocessors,
        n=workload.n,
        repeats=2,
        number=1,
        min_repeat_ms=10,
        seed=workload.seed,
        slow_threshold_ms=1e9,
    )
    rng = np.random.default_rng(workload.seed)
    for index in range(10):
        result, build_time = measure(
            random_position(rng, layout), schedule, layout, config
        )
        if result.status != "ok":
            raise RuntimeError(f"Smoke schedule {index} failed: {result.status}")
        print(
            f"{index + 1}/10: {result.median_ms:.4f} ms "
            f"(build {build_time:.3f} s)"
        )


if __name__ == "__main__":
    main()
