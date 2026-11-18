
"""Evaluation metrics for ranking measured schedules."""
from __future__ import annotations


import numpy as np
from scipy.stats import spearmanr


def spearman(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return rank correlation, or NaN when the inputs have no rank variance."""
    value = float(spearmanr(y_true, y_score).statistic)
    return value


def top_k_latency_ratio(y_true: np.ndarray, ranking: np.ndarray, k: int) -> float:
    """Mean latency of the selected top-k divided by the best measured latency."""
    y_true = np.asarray(y_true, dtype=float)
    selected = np.asarray(ranking, dtype=int)[:k]
    if selected.size == 0:
        raise ValueError("ranking must contain at least one candidate")
    return float(np.mean(y_true[selected]) / np.min(y_true))


def fastest_decile_spearman(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return rank correlation among the fastest 10% by measured latency."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_score must be one-dimensional arrays of equal size")
    if y_true.size < 2:
        return float("nan")
    count = max(2, int(np.ceil(y_true.size * 0.1)))
    fastest = np.argsort(y_true, kind="stable")[:count]
    return spearman(y_true[fastest], y_score[fastest])


def top_k_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    true_k: int = 10,
    predicted_k: int = 30,
) -> float:
    """Return the fraction of true top-k schedules found in the predicted top-k."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_score must be one-dimensional arrays of equal size")
    if true_k < 1 or predicted_k < 1:
        raise ValueError("true_k and predicted_k must be positive")
    if y_true.size == 0:
        raise ValueError("at least one observation is required")
    actual_count = min(true_k, y_true.size)
    predicted_count = min(predicted_k, y_true.size)
    actual = set(np.argsort(y_true, kind="stable")[:actual_count])
    predicted = set(np.argsort(y_score, kind="stable")[:predicted_count])
    return len(actual & predicted) / actual_count


def ranking_metrics(
    y_true: np.ndarray,
    predicted_latency: np.ndarray,
) -> dict[str, float]:
    """Evaluate one latency predictor with the study's shared ranking metrics."""
    y_true = np.asarray(y_true, dtype=float)
    predicted_latency = np.asarray(predicted_latency, dtype=float)
    if (
        y_true.ndim != 1
        or predicted_latency.ndim != 1
        or y_true.shape != predicted_latency.shape
        or y_true.size == 0
        or not np.all(np.isfinite(y_true))
        or not np.all(np.isfinite(predicted_latency))
        or np.any(y_true <= 0)
        or np.any(predicted_latency <= 0)
    ):
        raise ValueError("latencies must be equal-length, positive finite vectors")
    ranking = np.argsort(predicted_latency, kind="stable")
    values = {
        "spearman": spearman(y_true, predicted_latency),
        "fastest_decile_spearman": fastest_decile_spearman(
            y_true, predicted_latency
        ),
        "top_10_recall_in_30": top_k_recall(y_true, predicted_latency, 10, 30),
    }
    for k in (1, 5, 10):
        values[f"top_{k}_latency_ratio"] = top_k_latency_ratio(
            y_true, ranking, min(k, y_true.size)
        )
    return values


