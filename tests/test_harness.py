import hashlib
from types import SimpleNamespace

import numpy as np
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply

import dvcm.harness as harness
from dvcm.encoding import decisions_to_position, describe_space
from dvcm.config import PRIMARY_TARGET, load_target_config, load_workload_config
from dvcm.harness import (
    HarnessConfig,
    _schedule_from_position,
    build_from_position,
    make_matmul,
    time_module,
)
from dvcm.rpc import RPCExecutor


def _first_design_space():
    target_config = load_target_config(PRIMARY_TARGET)
    workload = load_workload_config()
    target = target_config.make_tvm_target()
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
    schedule = context.generate_design_space()[0]
    layout = describe_space(schedule.trace)
    layout.update(target_config.feature_settings())
    return schedule, layout, target, target_config, workload


def test_unroll_index_changes_postprocessed_program():
    schedule, layout, target, target_config, workload = _first_design_space()
    position = decisions_to_position(schedule.trace, layout).astype(int)
    categorical = layout["categoricals"][0]
    first = position.copy()
    second = position.copy()
    first[categorical["index"]] = 0
    second[categorical["index"]] = 1
    config = HarnessConfig(
        target=target,
        postprocessors=target_config.postprocessors,
        n=workload.n,
        repeats=workload.measurement["repeats"],
        number=workload.measurement["number"],
        min_repeat_ms=workload.measurement["min_repeat_ms"],
        seed=workload.seed,
        slow_threshold_ms=1e9,
    )
    first_tir = _schedule_from_position(first, schedule, layout, config).mod.script()
    second_tir = _schedule_from_position(second, schedule, layout, config).mod.script()
    first_hash = hashlib.sha256(first_tir.encode()).hexdigest()
    second_hash = hashlib.sha256(second_tir.encode()).hexdigest()
    assert first_hash != second_hash, (
        "Different unroll candidates produced identical TIR; "
        "the MetaSchedule post-processing is not being applied correctly."
    )


def test_known_good_schedule_cross_compiles_for_the_nano():
    schedule, layout, target, target_config, workload = _first_design_space()
    position = decisions_to_position(schedule.trace, layout)
    config = HarnessConfig(
        target=target,
        postprocessors=target_config.postprocessors,
        n=workload.n,
        repeats=2,
        number=1,
        min_repeat_ms=0,
        seed=workload.seed,
        slow_threshold_ms=1e9,
    )
    library, status = build_from_position(position, schedule, layout, config)
    assert status == "ok"
    assert library is not None


def test_rpc_timing_probes_once_then_times_the_repeats(monkeypatch):
    reference = np.zeros((1, 1), dtype=np.float32)

    class Output:
        def copyfrom(self, value):
            pass

        def numpy(self):
            return reference

    class Library:
        def __getitem__(self, name):
            assert name == "main"
            return lambda *args: None

        def time_evaluator(self, name, device, repeat, number, min_repeat_ms):
            assert name == "main"
            if repeat == 1:
                return lambda *args: SimpleNamespace(results=(0.001,))

            def evaluate(*args):
                events.append("evaluator")
                return SimpleNamespace(results=(0.002, 0.004))

            return evaluate

    events = []
    monkeypatch.setattr(harness, "_zero_output", lambda output, config: None)
    class Session:
        def cpu(self):
            return "remote_cpu"

    config = HarnessConfig(
        target="llvm",
        postprocessors="llvm",
        n=1,
        repeats=2,
        number=1,
        min_repeat_ms=20,
        seed=0,
        slow_threshold_ms=float("inf"),
    )

    executor = RPCExecutor(Session())
    executor.prepare_inputs = lambda n, seed: (object(), object(), Output(), reference)
    result = time_module(
        Library(),
        config,
        rpc_executor=executor,
    )

    assert events == ["evaluator"]
    assert result.repeats_ms == (2.0, 4.0)
