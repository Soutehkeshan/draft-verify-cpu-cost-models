"""The confirmation-set sampler and evaluation helpers."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from dvcm.encoding import is_valid_position

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(f"confirm_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _layout():
    return {
        "dim": 11,
        "tiles": [
            {"n": 4, "slice": slice(0, 4), "total_exp": 9, "max_inner_exp": 5},
            {"n": 4, "slice": slice(4, 8), "total_exp": 9, "max_inner_exp": 5},
            {"n": 2, "slice": slice(8, 10), "total_exp": 9, "max_inner_exp": 5},
        ],
        "categoricals": [{"index": 10, "candidates": [0, 16, 64, 512]}],
    }


def test_new_schedules_are_valid_unique_and_never_in_the_original_pool():
    sampler = _script("sample_confirmation_pool")
    layouts = [_layout(), _layout()]
    first = sampler.sample_new(layouts, set(), 300, seed=0)
    exclude = {(s, tuple(int(v) for v in p)) for s, p in first}
    second = sampler.sample_new(layouts, exclude, 300, seed=0)  # same seed: would repeat without the exclusion
    keys = [(s, tuple(int(v) for v in p)) for s, p in second]
    assert len(set(keys)) == 300
    assert not set(keys) & exclude
    assert all(is_valid_position(p, layouts[s]) for s, p in second)


def test_top_one_percent_recall_rewards_finding_the_fastest_schedules():
    evaluate = _script("evaluate_confirmation")
    truth = np.arange(1, 1001, dtype=float)  # schedule i takes i ms
    perfect = evaluate.confirmation_metrics(truth, truth)
    reversed_ = evaluate.confirmation_metrics(truth, truth[::-1].copy())
    assert perfect["top_1pct_recall_in_3pct"] == 1.0 and perfect["spearman"] == pytest.approx(1.0)
    assert reversed_["top_1pct_recall_in_3pct"] == 0.0
    assert set(perfect) == set(evaluate.METRICS)


def test_a_gain_is_confirmed_only_when_its_range_is_on_the_better_side_of_zero():
    evaluate = _script("evaluate_confirmation")
    assert evaluate.confirmed({"paired_bootstrap_ci_low": 0.001, "paired_bootstrap_ci_high": 0.01}, "spearman")
    assert not evaluate.confirmed({"paired_bootstrap_ci_low": -0.001, "paired_bootstrap_ci_high": 0.01}, "spearman")
    assert evaluate.confirmed({"paired_bootstrap_ci_low": -0.2, "paired_bootstrap_ci_high": -0.01}, "top_1_latency_ratio")
    assert not evaluate.confirmed({"paired_bootstrap_ci_low": 0.01, "paired_bootstrap_ci_high": 0.2}, "top_1_latency_ratio")


def test_a_perfect_drafter_keeps_every_target_and_random_keeps_its_share():
    evaluate = _script("evaluate_confirmation")
    truth = np.arange(1, 2001, dtype=float)
    rows = evaluate.draft_rows("pooled", truth, {"perfect": truth})
    perfect = [r for r in rows if r["drafter"] == "perfect"]
    assert all(r["share_kept"] == 1.0 for r in perfect)
    random = [r for r in rows if r["drafter"] == "random (expected)" and r["draft_fraction"] == 0.04]
    assert random[0]["share_kept"] == pytest.approx(0.04)
