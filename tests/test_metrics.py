import numpy as np
import pytest

from dvcm.metrics import ranking_metrics


def test_ranking_metrics_use_same_order_and_oracle_ratio():
    measured = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = np.array([1.0, 2.0, 3.0, 4.0])

    scores = ranking_metrics(measured, predictions)

    assert scores["spearman"] == 1.0
    assert scores["top_1_latency_ratio"] == 1.0
    assert scores["top_5_latency_ratio"] == np.mean(measured) / measured.min()
    assert scores["top_10_latency_ratio"] == np.mean(measured) / measured.min()
    assert scores["top_10_recall_in_30"] == 1.0


def test_ranking_metrics_reject_invalid_latency_vectors():
    with pytest.raises(ValueError, match="equal-length"):
        ranking_metrics(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="positive finite"):
        ranking_metrics(np.array([0.0]), np.array([1.0]))
