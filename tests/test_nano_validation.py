from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest

from dvcm.config import load_target_config, load_workload_config
from dvcm.harness import MeasurementResult, TimingResult
from scripts.validate_nano_schedules import _sample_positions


def test_nano_validation_sampler_covers_categories_and_tile_edges():
    layout = {
        "dim": 3,
        "tiles": [
            {
                "slice": slice(0, 2),
                "valid_exponents": [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)],
            }
        ],
        "categoricals": [{"index": 2, "candidates": [0, 16, 64, 512]}],
    }
    base = np.array([2, 2, 0])

    sample = _sample_positions(
        base, layout, np.random.default_rng(42), count=12
    )
    repeated = _sample_positions(
        base, layout, np.random.default_rng(42), count=12
    )

    positions = [tuple(position) for _, position in sample]
    assert positions == [tuple(position) for _, position in repeated]
    assert len(set(positions)) == 12
    assert {position[2] for position in positions} == {0, 1, 2, 3}
    assert {(position[0], position[1]) for position in positions} >= {
        (0, 4),
        (2, 2),
        (4, 0),
    }


def test_nano_validation_sampler_rejects_oversized_sample():
    layout = {
        "dim": 2,
        "tiles": [{"slice": slice(0, 2), "valid_exponents": [(0, 1), (1, 0)]}],
        "categoricals": [],
    }

    with pytest.raises(ValueError, match="exceeds"):
        _sample_positions(
            np.array([0, 1]), layout, np.random.default_rng(42), count=3
        )


def test_collection_row_preserves_remote_timing_phases():
    root = Path(__file__).resolve().parents[1]
    collector = run_path(
        str(root / "scripts" / "01_collect.py"), run_name="collector_test"
    )
    measurement = MeasurementResult(
        timing=TimingResult((2.0, 2.2), 2.1, 0.1, "ok"),
        build_time_s=0.3,
        export_time_s=0.4,
        upload_time_s=0.5,
        link_load_time_s=0.6,
    )

    row = collector["_record"](7, 2, np.array([1, 2]), measurement, 3)

    assert row["id"] == 7
    assert row["sketch_id"] == 2
    assert row["status"] == "ok"
    assert row["repeats_ms"] == [2.0, 2.2]
    assert row["build_time_s"] == 0.3
    assert row["export_time_s"] == 0.4
    assert row["upload_time_s"] == 0.5
    assert row["link_load_time_s"] == 0.6

    target_config = load_target_config("jetson_nano", root)
    workload = load_workload_config(root=root)
    executor = SimpleNamespace(
        input_prepare_time_s=0.7,
        input_upload_time_s=0.8,
    )
    metadata = collector["_execution_metadata"](
        target_config,
        workload,
        target_config.make_tvm_target(),
        executor,
        "127.0.0.1",
        9090,
    )

    assert metadata["target_id"] == "jetson_nano"
    assert metadata["target_profile"]["compiler_target"]["mcpu"] == "cortex-a57"
    assert (
        metadata["target_profile"]["compiler_target"]["mtriple"]
        == "aarch64-linux-gnu"
    )
    assert "cortex-a57" in metadata["target"]
    assert metadata["execution_backend"] == "rpc"
    assert metadata["rpc_host"] == "127.0.0.1"
    assert metadata["rpc_port"] == 9090
    assert metadata["rpc_input_prepare_time_s"] == 0.7
    assert metadata["rpc_input_upload_time_s"] == 0.8
