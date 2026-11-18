from dataclasses import replace

from dvcm.features import features
import numpy as np
import pytest

from dvcm.config import PRIMARY_TARGET, load_target_config, load_workload_config


def test_nano_profile_uses_explicit_llvm_rules_and_its_own_hardware_values():
    assert load_target_config(PRIMARY_TARGET).target_id == "jetson_nano"
    nano = load_target_config("jetson_nano")
    assert nano.cpu == "ARM Cortex-A57"
    assert nano.num_cores == 4
    assert nano.paths()[0].as_posix().endswith("data/jetson_nano")
    assert nano.compiler_target == {
        "kind": "llvm",
        "mtriple": "aarch64-linux-gnu",
        "mcpu": "cortex-a57",
        "num-cores": 4,
    }
    nano_target = nano.make_tvm_target()
    assert nano_target.kind.name == "llvm"
    assert nano_target.attrs["mtriple"] == "aarch64-linux-gnu"
    assert nano_target.attrs["mcpu"] == "cortex-a57"
    # Explicit llvm rules; "from-target" would pick ARM-specific rules for this target.
    assert (nano.schedule_rules, nano.postprocessors) == ("llvm", "llvm")
    assert nano.feature_settings() == {"vector_lanes": 4, "num_cores": 4, "vector_registers": 32}
    nano.require_benchmarking_profile()
    with pytest.raises(RuntimeError, match="not configured"):
        replace(nano, schedule_rules=None).require_benchmarking_profile()
    assert nano.analytic_constants() == {
        "peak_flops": 4 * 1.479e9 * 8,
        "bandwidth_bytes_s": 1e10,
        "l1_bytes": 32768,
        "l2_bytes": 2097152,
        "lane_width": 4,
        "num_cores": 4,
        "vector_registers": 32,
    }
    for name in ("peak_flops", "bandwidth_bytes_s"):
        assert nano.notes[name].startswith("initial guess, fitted on TRAIN only")
    assert nano.feature_settings() == {
        "vector_lanes": 4,
        "num_cores": 4,
        "vector_registers": 32,
    }
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
