from dataclasses import replace
import json

import pytest
import pandas as pd

from dvcm.config import load_target_config
from dvcm.study import (
    load_layouts,
    make_splits,
    restrict_splits,
    validate_splits,
)


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


def test_split_generation_is_seeded_stratified_and_disjoint():
    frame = pd.DataFrame(
        [
            {"id": sketch * 20 + index, "sketch_id": sketch}
            for sketch in range(3)
            for index in range(20)
        ]
    )

    splits = make_splits(frame, seed=0)

    assert splits == make_splits(frame, seed=0)
    validate_splits(splits, set(frame["id"]))
    for partition in ("train", "validation", "test"):
        counts = frame[frame["id"].isin(splits[partition])].groupby("sketch_id").size()
        assert counts.nunique() == 1


def test_restrict_splits_keeps_roles_order_and_nesting():
    source = {
        "seed": 0,
        "train": [1, 2, 3, 4],
        "validation": [5, 6],
        "test": [8, 7],
        "nested_train": {"2": [3, 1], "3": [3, 1, 4], "all": [3, 1, 4, 2]},
    }
    assert restrict_splits(source, set(range(1, 9))) == source

    restricted = restrict_splits(source, {1, 2, 4, 5, 7, 8})

    assert restricted["train"] == [1, 2, 4]
    assert restricted["validation"] == [5]
    assert restricted["test"] == [8, 7]
    assert restricted["nested_train"] == {"2": [1], "3": [1, 4], "all": [1, 4, 2]}
    validate_splits(restricted, {1, 2, 4, 5, 7, 8})


def test_target_binding_does_not_reuse_legacy_layout_hardware_values(tmp_path):
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
