"""scripts/07_transfer.py: split-role and leakage guards, out-of-fold M2 predictions, selection."""
import importlib.util
import itertools
from pathlib import Path

import pytest

from dvcm.learned import HYPERPARAMETER_GRID

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "train": [1, 2, 3, 4, 5, 6],
    "validation": [7, 8],
    "test": [9, 10],
    "nested_train": {"2": [3, 1], "all": [3, 1, 6, 2, 5, 4]},
}


def _transfer():
    spec = importlib.util.spec_from_file_location("transfer_test", ROOT / "scripts" / "07_transfer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_shared_grid_is_the_training_grid():
    assert list(HYPERPARAMETER_GRID) == list(itertools.product((3, 6), (0.03, 0.1), (100, 300)))


def test_transfer_requires_both_targets_to_share_the_split_roles():
    transfer = _transfer()
    assert transfer.m2_training_ids(SPLITS, SPLITS) == [1, 2, 3, 4, 5, 6]
    moved = {**SPLITS, "train": [1, 2, 3, 4, 5, 9], "test": [6, 10]}
    with pytest.raises(ValueError, match="share"):
        transfer.m2_training_ids(moved, SPLITS)
    reordered = {**SPLITS, "nested_train": {"2": [1, 3], "all": SPLITS["nested_train"]["all"]}}
    with pytest.raises(ValueError, match="share"):
        transfer.m2_training_ids(reordered, SPLITS)


def test_m2_rows_of_validation_or_test_ids_are_never_trained_on():
    transfer = _transfer()
    transfer.require_disjoint([1, 2, 3], SPLITS)
    with pytest.raises(ValueError, match="VALIDATION/TEST"):
        transfer.require_disjoint([1, 2, 9], SPLITS)
    with pytest.raises(ValueError, match="VALIDATION/TEST"):
        transfer.require_disjoint([7], SPLITS)


def test_out_of_fold_predictions_never_come_from_a_model_trained_on_that_row():
    transfer = _transfer()
    ids = list(range(1, 23))
    folds = transfer.oof_folds(ids, n_folds=5, seed=0)
    assert set(folds) == set(ids) and set(folds.values()) == set(range(5))
    sizes = [list(folds.values()).count(fold) for fold in range(5)]
    assert max(sizes) - min(sizes) <= 1
    assert folds == transfer.oof_folds(ids, n_folds=5, seed=0)
    calls = []

    def fit_predict(fold, train_ids, predict_ids):
        calls.append((fold, set(train_ids), set(predict_ids)))
        return {sample_id: float(sample_id) for sample_id in predict_ids}

    predictions = transfer.out_of_fold_predictions(ids, folds, fit_predict)

    assert predictions == {sample_id: float(sample_id) for sample_id in ids}
    assert [fold for fold, _, _ in calls] == list(range(5))
    for fold, train_ids, predict_ids in calls:
        assert not train_ids & predict_ids
        assert train_ids | predict_ids == set(ids)
        assert {folds[sample_id] for sample_id in predict_ids} == {fold}


def test_settings_are_chosen_by_validation_log_rmse_and_the_first_tie_wins():
    transfer = _transfer()
    grid = [(3, 0.03, 100), (3, 0.1, 100), (6, 0.1, 300)]
    scores = {grid[0]: 0.30, grid[1]: 0.20, grid[2]: 0.20}
    best, results = transfer.select_config(grid, lambda config: scores[config])
    assert best == {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100,
                    "validation_log_rmse": 0.20}
    assert [result["validation_log_rmse"] for result in results] == [0.30, 0.20, 0.20]
