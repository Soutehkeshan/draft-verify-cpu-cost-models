"""scripts/check_shuffled_label_null.py: more global label permutations, made as 04_evaluate makes them."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "shuffle_check_test", ROOT / "scripts" / "check_shuffled_label_null.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_permutations_are_the_ones_the_evaluation_draws():
    module = _module()
    labels = np.arange(12.0)
    for seed in (0, 7, 19, 119):
        expected = np.random.default_rng(seed).permutation(labels)  # 04_evaluate.py, global mode
        assert np.array_equal(module.permuted_labels(labels, seed), expected)
    assert np.array_equal(labels, np.arange(12.0))  # the input is not shuffled in place


def test_summary_states_the_mean_in_standard_errors():
    module = _module()
    values = np.array([0.1, -0.1, 0.2, 0.0])
    summary = module.summarize(values)
    standard_error = np.std(values, ddof=1) / np.sqrt(values.size)
    assert summary["n_permutations"] == 4
    assert summary["mean"] == pytest.approx(values.mean())
    assert summary["standard_error"] == pytest.approx(standard_error)
    assert summary["mean_over_standard_error"] == pytest.approx(values.mean() / standard_error)
    assert (summary["min"], summary["max"]) == (-0.1, 0.2)
