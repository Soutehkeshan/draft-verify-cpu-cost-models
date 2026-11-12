
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
