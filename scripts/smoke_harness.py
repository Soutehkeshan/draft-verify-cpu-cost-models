#!/usr/bin/env python3
"""Compile and measure ten seeded schedules as a fast integration check."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tvm
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.encoding import describe_space, random_position
from dvcm.harness import HarnessConfig, make_matmul, measure


def main() -> None:
    target = tvm.target.Target(
        {"kind": "llvm", "mcpu": "apple-m2", "num-cores": 4}
    )
    context = ms.TuneContext(
        mod=make_matmul(512),
        target=target,
        space_generator=PostOrderApply(sch_rules="llvm", postprocs="llvm"),
        rand_state=0,
        num_threads=4,
    )
    schedule = context.generate_design_space()[0]
    layout = describe_space(schedule.trace)
    config = HarnessConfig(
        target=target,
        repeats=2,
        number=1,
        min_repeat_ms=10,
        seed=0,
        slow_threshold_ms=1e9,
    )
    rng = np.random.default_rng(0)
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
