import pytest

from dvcm.study import validate_splits


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
