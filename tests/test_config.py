from dvcm.features import features
import numpy as np
import pytest

from dvcm.config import LEGACY_TARGET, load_target_config, load_workload_config


def test_target_profiles_keep_m2_compatibility_and_nano_unknowns_explicit():
    m2 = load_target_config(LEGACY_TARGET)
    assert m2.compiler_target == {
        "kind": "llvm",
        "mcpu": "apple-m2",
        "num-cores": 4,
    }
    assert m2.feature_settings() == {
        "vector_lanes": 4,
        "num_cores": 4,
        "vector_registers": 32,
    }
    assert m2.paths()[1].as_posix().endswith("results/apple_m2")

    nano = load_target_config("jetson_nano")
    assert nano.cpu == "ARM Cortex-A57"
    assert nano.num_cores == 4
    assert nano.paths()[0].as_posix().endswith("data/jetson_nano")
    assert nano.compiler_target is None
    with pytest.raises(RuntimeError, match="not configured"):
        nano.require_benchmarking_profile()
    with pytest.raises(ValueError, match="analytical constants"):
        nano.analytic_constants()
    with pytest.raises(ValueError, match="feature settings"):
        nano.feature_settings()
    position = np.array([9, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0], dtype=float)
    layout = {
        "dim": 11,
        "tiles": [
            {"n": 4, "slice": [0, 4], "total_exp": 9, "max_inner_exp": 6},
            {"n": 4, "slice": [4, 8], "total_exp": 9, "max_inner_exp": 6},
            {"n": 2, "slice": [8, 10], "total_exp": 9, "max_inner_exp": 6},
        ],
        "categoricals": [{"index": 10, "candidates": [0, 16, 64, 512]}],
        "parallel_tile_levels": {"i": 2, "j": 2},
    }
    assert np.isfinite(features(position, 0, layout, include_derived=False)).all()
    with pytest.raises(ValueError, match="Target feature settings"):
        features(position, 0, layout)


def test_workload_profile_describes_single_fp32_matmul():
    workload = load_workload_config()
    assert (workload.m, workload.n, workload.k, workload.dtype) == (
        512,
        512,
        512,
        "float32",
    )
    assert workload.seed == 0
