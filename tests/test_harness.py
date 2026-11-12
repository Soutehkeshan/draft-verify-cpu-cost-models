import hashlib

import numpy as np
import tvm
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply

from dvcm.encoding import decisions_to_position, describe_space
from dvcm.harness import (
    HarnessConfig,
    _schedule_from_position,
    build_from_position,
    make_matmul,
    time_module,
)


def _first_design_space():
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
    return schedule, describe_space(schedule.trace), target


def test_unroll_index_changes_postprocessed_program():
    schedule, layout, target = _first_design_space()
    position = decisions_to_position(schedule.trace, layout).astype(int)
    categorical = layout["categoricals"][0]
    first = position.copy()
    second = position.copy()
    first[categorical["index"]] = 0
    second[categorical["index"]] = 1
    config = HarnessConfig(target=target)
    first_tir = _schedule_from_position(first, schedule, layout, config).mod.script()
    second_tir = _schedule_from_position(second, schedule, layout, config).mod.script()
    first_hash = hashlib.sha256(first_tir.encode()).hexdigest()
    second_hash = hashlib.sha256(second_tir.encode()).hexdigest()
    assert first_hash != second_hash, (
        "Different unroll candidates produced identical TIR; "
        "the MetaSchedule post-processing is not being applied correctly."
    )


def test_known_good_schedule_builds_and_passes_correctness():
    schedule, layout, target = _first_design_space()
    position = decisions_to_position(schedule.trace, layout)
    config = HarnessConfig(
        target=target,
        repeats=2,
        number=1,
        min_repeat_ms=0,
        slow_threshold_ms=1e9,
    )
    library, status = build_from_position(position, schedule, layout, config)
    assert status == "ok"
    assert library is not None
    result = time_module(library, config)
    assert result.status == "ok"
    assert len(result.repeats_ms) == config.repeats
    assert np.isfinite(result.median_ms)
