import numpy as np
import tvm
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply

from dvcm.encoding import (
    describe_space,
    decisions_to_position,
    random_position,
    repair,
    repair_tile,
    position_to_decisions,
)
from dvcm.features import feature_names, features
from dvcm.harness import make_matmul
from dvcm.study import add_schedule_if_new


def _design_spaces():
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
    return context.generate_design_space()


def test_repair_is_idempotent_for_random_positions():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        position = rng.normal(3.0, 5.0, size=11)
        repaired = repair(position)
        np.testing.assert_array_equal(repair(repaired), repaired)


def test_random_positions_are_valid_and_deduplicate():
    rng = np.random.default_rng(0)
    seen = set()
    for sketch_id, schedule in enumerate(_design_spaces()):
        layout = describe_space(schedule.trace)
        for _ in range(200):
            position = random_position(rng, layout).astype(int)
            for tile in layout["tiles"]:
                exponents = position[tile["slice"]]
                assert exponents.sum() == tile["total_exp"]
                assert exponents[-1] <= tile["max_inner_exp"]
            categorical = layout["categoricals"][0]
            assert 0 <= position[categorical["index"]] < len(categorical["candidates"])
            assert add_schedule_if_new(seen, sketch_id, position)
    first_sketch, first_position = 0, np.zeros(11, dtype=int)
    assert add_schedule_if_new(seen, first_sketch, first_position)
    assert not add_schedule_if_new(seen, first_sketch, first_position)


def test_position_trace_round_trip_and_features():
    rng = np.random.default_rng(0)
    spaces = _design_spaces()
    for _ in range(200):
        sketch_id = int(rng.integers(len(spaces)))
        schedule = spaces[sketch_id]
        layout = describe_space(schedule.trace)
        position = random_position(rng, layout)
        updated_trace = position_to_decisions(position, schedule, layout)
        round_trip = decisions_to_position(updated_trace, layout)
        np.testing.assert_array_equal(round_trip, repair(position, layout))

        full_features = features(position, sketch_id, layout)
        base_features = features(position, sketch_id, layout, include_derived=False)
        assert len(feature_names(layout)) == len(full_features)
        assert len(feature_names(layout, include_derived=False)) == len(base_features)
        assert np.all(np.isfinite(full_features))
        assert np.all(np.isfinite(base_features))


def test_layout_records_sketch_specific_parallel_loop_prefix():
    actual = [
        describe_space(schedule.trace)["parallel_tile_levels"]
        for schedule in _design_spaces()
    ]
    assert actual == [
        {"i": 2, "j": 2},
        {"i": 1, "j": 1},
        {"i": 2, "j": 2},
    ]


def test_repair_tile_enforces_composition():
    repaired = repair_tile(np.array([100.0, -5.0, 0.1, 0.0]), total=9, max_inner=5)
    assert repaired.sum() == 9
    assert repaired[-1] <= 5
