import numpy as np

from dvcm.config import LEGACY_TARGET, load_target_config
from dvcm.analytic import AnalyticConstants, Penalties, latency_seconds, penalty_terms
from dvcm.features import (
    ALL_DERIVED_NAMES,
    DERIVED_NAMES,
    bind_target_settings,
    derived,
    feature_names,
    features,
)
from dvcm.metrics import fastest_decile_spearman, top_k_recall


def _layout():
    return {
        "dim": 11,
        "tiles": [
            {"n": 4, "slice": [0, 4], "total_exp": 9, "max_inner_exp": 6},
            {"n": 4, "slice": [4, 8], "total_exp": 9, "max_inner_exp": 6},
            {"n": 2, "slice": [8, 10], "total_exp": 9, "max_inner_exp": 6},
        ],
        "categoricals": [{"index": 10, "candidates": [0, 16, 64, 512]}],
        "parallel_tile_levels": {"i": 2, "j": 2},
        "vector_lanes": 4,
        "num_cores": 4,
        "vector_registers": 32,
    }


def _positions():
    tiny_tile = np.array([9, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0])
    large_tile = np.array([0, 0, 3, 6, 0, 0, 3, 6, 3, 6, 2])
    return tiny_tile, large_tile


def _m2_constants(**overrides):
    values = load_target_config(LEGACY_TARGET).analytic_constants()
    values.update(overrides)
    return AnalyticConstants(**values)


def test_nano_constants_and_feature_binding_come_from_the_nano_profile():
    nano = load_target_config("jetson_nano")
    constants = AnalyticConstants.from_target(nano)
    assert (constants.l1_bytes, constants.l2_bytes) == (32768, 2097152)
    assert (constants.lane_width, constants.num_cores, constants.vector_registers) == (4, 4, 32)
    m2 = AnalyticConstants.from_target(load_target_config(LEGACY_TARGET))
    assert constants.peak_flops != m2.peak_flops
    assert constants.bandwidth_bytes_s != m2.bandwidth_bytes_s

    layout = {**_layout(), "vector_lanes": 8, "num_cores": 8, "vector_registers": 16}
    bound = bind_target_settings(layout, nano)
    assert (bound["vector_lanes"], bound["num_cores"], bound["vector_registers"]) == (4, 4, 32)
    assert layout["vector_lanes"] == 8  # input layout is not mutated
    # A 2 MiB L2 penalises the large tile that fits the M2's 16 MiB L2.
    _, large_tile = _positions()
    assert penalty_terms(large_tile, bound, constants)["cache_l2"] < 1.0
    assert penalty_terms(large_tile, _layout(), m2)["cache_l2"] == 1.0


def test_flops_and_bytes_are_correct_for_tiny_and_large_tiles():
    layout = _layout()
    tiny_tile, large_tile = _positions()
    for position in (tiny_tile, large_tile):
        descriptors = derived(position, layout)
        assert descriptors["flop_count"] == 2 * 512**3
        assert descriptors["tile_level_0_bytes"] == 3 * 4 * 512**2
        assert np.isfinite(list(descriptors.values())).all()

    tiny = derived(tiny_tile, layout)
    large = derived(large_tile, layout)
    assert tiny["bytes_moved"] == 4 * (2 * 512**3 + 512**2)
    assert large["bytes_moved"] == 3 * 4 * 512**2
    assert tiny["tile_level_2_bytes"] == 12
    assert tiny["tile_level_3_bytes"] == 12
    assert large["tile_level_1_bytes"] == 3 * 4 * 512**2
    assert large["tile_level_2_bytes"] == 4 * (2 * 512 * 64 + 512**2)
    assert large["tile_level_3_bytes"] == 4 * (64**2 + 2 * 64)


def test_model_features_match_names_and_exclude_known_invariants():
    layout = _layout()
    tiny_tile, _ = _positions()
    values = features(tiny_tile, 0, layout)
    assert len(feature_names(layout)) == len(values)
    assert len(feature_names(layout, include_derived=False)) == len(
        features(tiny_tile, 0, layout, include_derived=False)
    )
    assert len(DERIVED_NAMES) == len(set(DERIVED_NAMES))
    assert set(ALL_DERIVED_NAMES) >= set(DERIVED_NAMES)
    assert "flop_count" not in DERIVED_NAMES
    assert "tile_level_0_bytes" not in DERIVED_NAMES
    assert "tile_footprint_bytes" not in ALL_DERIVED_NAMES
    assert "unroll_value" not in DERIVED_NAMES
    assert "tile_level_1_bytes" not in DERIVED_NAMES
    assert "register_utilisation" not in DERIVED_NAMES
    assert np.isfinite(values).all()


def test_utilization_and_penalty_terms_are_bounded():
    layout = _layout()
    tiny_tile, large_tile = _positions()
    tiny = derived(tiny_tile, layout)
    large = derived(large_tile, layout)
    assert tiny["vector_lane_utilisation"] == 0.25
    assert large["vector_lane_utilisation"] == 1.0
    assert tiny["parallel_utilisation"] == 1.0
    assert large["parallel_utilisation"] == 0.25
    assert tiny["register_utilisation"] == 1.0
    assert large["register_utilisation"] == 0.03125

    constants = _m2_constants()
    for position in (tiny_tile, large_tile):
        factors = penalty_terms(position, layout, constants)
        assert set(factors) == {
            "vector",
            "parallel",
            "register",
            "cache_l1",
            "cache_l2",
        }
        assert np.isfinite(list(factors.values())).all()
        assert all(0.0 <= value <= 1.0 for value in factors.values())
        assert np.isfinite(latency_seconds(position, layout, constants))

    ordinary = penalty_terms(tiny_tile, layout, constants)
    assert all(ordinary[name] == 1.0 for name in ("cache_l1", "cache_l2"))
    constrained = penalty_terms(
        tiny_tile,
        layout,
        _m2_constants(l1_bytes=1, l2_bytes=1),
    )
    assert all(constrained[name] < 1.0 for name in ("cache_l1", "cache_l2"))

    disabled = penalty_terms(
        large_tile,
        layout,
        constants,
        penalties=Penalties(
            vector=False, parallel=False, cache=False, register=False
        ),
    )
    assert all(value == 1.0 for value in disabled.values())


def test_latency_is_compute_plus_memory_work_over_effective_peak():
    layout = _layout()
    tiny_tile, _ = _positions()
    constants = _m2_constants()
    descriptors = derived(tiny_tile, layout)
    factors = penalty_terms(tiny_tile, layout, constants)
    expected = descriptors["flop_count"] / (
        constants.peak_flops
        * factors["vector"]
        * factors["parallel"]
        * factors["register"]
    ) + descriptors["bytes_moved"] / (
        constants.bandwidth_bytes_s
        * factors["cache_l1"]
        * factors["cache_l2"]
    )
    assert latency_seconds(tiny_tile, layout, constants) == expected


def test_parallel_utilisation_accounts_for_partial_core_batches():
    layout = _layout()
    layout["num_cores"] = 3
    position = np.array([2, 0, 1, 6, 0, 0, 3, 6, 9, 0, 0])
    assert derived(position, layout)["parallel_utilisation"] == 2 / 3


def test_ranking_metrics_focus_on_fastest_schedules():
    actual = np.arange(1, 21, dtype=float)
    predicted = actual.copy()
    assert np.isclose(fastest_decile_spearman(actual, predicted), 1.0)
    assert top_k_recall(actual, predicted, true_k=3, predicted_k=5) == 1.0
    assert top_k_recall(actual, predicted[::-1], true_k=3, predicted_k=5) == 0.0
