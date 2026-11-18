from dataclasses import replace
import json

import pytest

from dvcm.config import load_target_config
from dvcm.study import collection_complete, load_layouts, validate_splits


def test_split_validation_requires_disjoint_exhaustive_partitions():
    splits = {
        "train": [1, 2],
        "validation": [3],
        "test": [4],
        "nested_train": {"1": [1], "2": [1, 2], "all": [1, 2]},
    }
    validate_splits(splits, {1, 2, 3, 4})

    overlapping = {**splits, "test": [2, 4]}
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(overlapping, {1, 2, 3, 4})


def test_split_validation_rejects_non_nested_training_subsets():
    splits = {
        "train": [1, 2, 3],
        "validation": [4],
        "test": [5],
        "nested_train": {"1": [1], "2": [2, 3], "all": [1, 2, 3]},
    }
    with pytest.raises(ValueError, match="not nested"):
        validate_splits(splits, {1, 2, 3, 4, 5})


def test_collection_complete_reads_the_collection_flag():
    assert collection_complete({"collection": {"collection_complete": True}})
    assert not collection_complete({"collection": {"collection_complete": False}})
    assert not collection_complete({"collection": {"state": "running"}})
    assert not collection_complete({})


def test_target_binding_takes_hardware_values_from_the_profile_only(tmp_path):
    layout = {
        "dim": 11,
        "tiles": [],
        "categoricals": [],
        "parallel_tile_levels": {"i": 2, "j": 2},
        "vector_lanes": 4,
        "num_cores": 4,
        "vector_registers": 32,
    }
    layout.update({"vector_lanes": 8, "num_cores": 8, "vector_registers": 16})
    (tmp_path / "layout_0.json").write_text(json.dumps(layout))
    nano = load_target_config("jetson_nano")

    nano_layout = load_layouts(tmp_path, nano)[0]

    assert (
        nano_layout["vector_lanes"],
        nano_layout["num_cores"],
        nano_layout["vector_registers"],
    ) == (4, 4, 32)

    unverified = replace(nano, simd_width_f32=None, vector_registers=None)
    unverified_layout = load_layouts(tmp_path, unverified)[0]

    assert unverified_layout["num_cores"] == 4
    assert "vector_lanes" not in unverified_layout
    assert "vector_registers" not in unverified_layout
