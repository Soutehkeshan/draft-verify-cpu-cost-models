
"""Evaluation metrics for ranking measured schedules."""
from __future__ import annotations

from collections.abc import Callable

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


def best_k(y_true: np.ndarray, ranking: np.ndarray, k: int) -> float:
    """Best latency among the top-k selected schedules, normalized by the oracle."""
    y_true = np.asarray(y_true, dtype=float)
    selected = np.asarray(ranking, dtype=int)[:k]
    if selected.size == 0:
        raise ValueError("ranking must contain at least one candidate")
    return float(np.min(y_true[selected]) / np.min(y_true))


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


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    seed: int = 0,
    samples: int = 2000,
) -> tuple[float, float]:
    """Percentile 95% bootstrap confidence interval over test observations."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    values = []
    for _ in range(samples):
        indexes = rng.integers(0, y_true.size, size=y_true.size)
        try:
            value = float(metric(y_true[indexes], y_score[indexes]))
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))
