"""scripts/03_fit_analytic.py: analytic constants refit on every nested TRAIN subset."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dvcm.analytic import AnalyticConstants, constants_from_payload, fit_constants, latency_seconds
from dvcm.config import LEGACY_TARGET, load_target_config
from test_models import _layout, _positions

ROOT = Path(__file__).resolve().parents[1]


def _fitter():
    spec = importlib.util.spec_from_file_location("fit_analytic_test", ROOT / "scripts" / "03_fit_analytic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log_rmse(frame, layouts, constants):
    predicted = [
        latency_seconds(np.asarray(v, dtype=float), layouts[int(s)], constants) * 1000
        for v, s in zip(frame["vector"], frame["sketch_id"])
    ]
    return float(np.sqrt(np.mean((np.log(predicted) - np.log(frame["median_ms"])) ** 2)))


def test_each_nested_subset_is_fit_on_its_own_rows_and_all_reuses_the_train_fit():
    layouts = {0: _layout()}
    initial = AnalyticConstants(**load_target_config(LEGACY_TARGET).analytic_constants())
    subset_truth = AnalyticConstants(**{**initial.to_dict(), "peak_flops": 2.5e11, "bandwidth_bytes_s": 8.0e10})
    other_truth = AnalyticConstants(**{**initial.to_dict(), "peak_flops": 2.0e10, "bandwidth_bytes_s": 5.0e9})
    positions = _positions()
    latencies = [
        latency_seconds(p, layouts[0], subset_truth if i < 3 else other_truth) * 1000
        for i, p in enumerate(positions)
    ]
    train = pd.DataFrame({"id": [0, 1, 2, 3, 4], "sketch_id": [0] * 5,
                          "vector": [p.astype(int).tolist() for p in positions], "median_ms": latencies})
    train_fit = fit_constants(positions, [layouts[0]] * 5, np.asarray(latencies), initial=initial)

    nested = _fitter().fit_nested_constants(
        train, {"3": [2, 0, 1], "all": [4, 3, 2, 1, 0]}, layouts, initial, train_fit
    )

    assert nested["all"]["constants"] == train_fit.to_dict()
    assert nested["all"]["training_samples"] == 5
    assert nested["3"]["training_samples"] == 3
    subset_fit = AnalyticConstants(**nested["3"]["constants"])
    subset_rows = train[train["id"] < 3]
    assert _log_rmse(subset_rows, layouts, subset_fit) < 1e-6
    assert _log_rmse(subset_rows, layouts, train_fit) > 1e-3
    assert nested["3"]["fit_multipliers"]["peak_flops"] == subset_fit.peak_flops / initial.peak_flops


def test_constants_payload_gives_hand_set_fitted_and_nested_constants():
    initial = AnalyticConstants(**load_target_config(LEGACY_TARGET).analytic_constants())
    fitted = AnalyticConstants(**{**initial.to_dict(), "peak_flops": 1.0e11})
    payload = {
        "hand_set_constants": initial.to_dict(),
        "constants": fitted.to_dict(),
        "nested_train_constants": {"all": {"constants": fitted.to_dict()},
                                   "100": {"constants": initial.to_dict()}},
    }
    hand_set, train_fit, nested = constants_from_payload(payload)
    assert (hand_set, train_fit) == (initial, fitted)
    assert nested == {"all": fitted, "100": initial}
    with pytest.raises(ValueError, match="nested_train_constants"):
        constants_from_payload({**payload, "nested_train_constants": {}})
