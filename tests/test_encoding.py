import numpy as np
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply

from dvcm.config import PRIMARY_TARGET, load_target_config, load_workload_config
from dvcm.encoding import (
    decisions_to_position,
    describe_space,
    is_valid_position,
    position_to_decisions,
    random_position,
)
from dvcm.features import feature_names, features
from dvcm.harness import make_matmul


def _design_spaces():
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
    return context.generate_design_space()


def _profiled_layout(schedule):
    layout = describe_space(schedule.trace)
    layout.update(load_target_config(PRIMARY_TARGET).feature_settings())
    return layout


def test_only_valid_positions_are_accepted():
    layout = _profiled_layout(_design_spaces()[0])
    position = random_position(np.random.default_rng(0), layout)
    assert is_valid_position(position, layout)
    i_tiles = layout["tiles"][0]["slice"]
    for broken in (
        np.where(np.arange(11) == i_tiles.start, position + 1, position),  # tile sum != 9
        np.where(np.arange(11) == 10, 4, position),  # unroll index out of range
        position + 0.5,  # not integers
        position[:10],  # wrong length
    ):
        assert not is_valid_position(broken, layout)


def test_random_positions_are_valid():
    rng = np.random.default_rng(0)
    for schedule in _design_spaces():
        layout = _profiled_layout(schedule)
        for _ in range(200):
            position = random_position(rng, layout).astype(int)
            for tile in layout["tiles"]:
                exponents = position[tile["slice"]]
                assert exponents.sum() == tile["total_exp"]
                assert exponents[-1] <= tile["max_inner_exp"]
            categorical = layout["categoricals"][0]
            assert 0 <= position[categorical["index"]] < len(categorical["candidates"])
            assert is_valid_position(position, layout)


def test_position_trace_round_trip_and_features():
    rng = np.random.default_rng(0)
    spaces = _design_spaces()
    for _ in range(200):
        sketch_id = int(rng.integers(len(spaces)))
        schedule = spaces[sketch_id]
        layout = _profiled_layout(schedule)
        position = random_position(rng, layout)
        updated_trace = position_to_decisions(position, schedule, layout)
        round_trip = decisions_to_position(updated_trace, layout)
        np.testing.assert_array_equal(round_trip, position)

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

