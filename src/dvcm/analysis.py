"""Bootstrap summaries, timing reliability and descriptive statistics for the result scripts."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from dvcm.metrics import ranking_metrics

REPORTED_METRICS = (
    "spearman",
    "fastest_decile_spearman",
    "top_10_recall_in_30",
    "top_1_latency_ratio",
    "top_5_latency_ratio",
)


def bootstrap_samples(size: int, n_resamples: int, seed: int) -> list[np.ndarray]:
    """TEST-row resamples, drawn one after another from one seeded generator."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, size, size=size) for _ in range(n_resamples)]


def resampled_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    seed: int,
    n_resamples: int,
    metrics: Sequence[str] = REPORTED_METRICS,
) -> np.ndarray:
    """ranking_metrics() of one predictor on each bootstrap sample (rows) per metric (columns)."""
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    values = np.empty((n_resamples, len(metrics)), dtype=float)
    for row, sample in enumerate(bootstrap_samples(len(y_true), n_resamples, seed)):
        scores = ranking_metrics(y_true[sample], prediction[sample])
        values[row] = [scores[name] for name in metrics]
    return values


def _resampled_task(task: tuple[Any, ...]) -> np.ndarray:
    return resampled_metrics(*task)


def resampled_metrics_parallel(
    tasks: Sequence[tuple[np.ndarray, np.ndarray, int]],
    n_resamples: int,
    metrics: Sequence[str] = REPORTED_METRICS,
    workers: int | None = None,
) -> list[np.ndarray]:
    """resampled_metrics() for (y_true, prediction, seed) tasks, in task order, across processes."""
    jobs = [(y_true, prediction, seed, n_resamples, tuple(metrics)) for y_true, prediction, seed in tasks]
    workers = min(len(jobs), workers or os.cpu_count() or 1)
    if workers <= 1:
        return [_resampled_task(job) for job in jobs]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(_resampled_task, jobs))


def _seed_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def paired_difference(
    reference_point: Sequence[dict[str, float]],
    candidate_point: Sequence[dict[str, float]],
    reference_resampled: Sequence[np.ndarray],
    candidate_resampled: Sequence[np.ndarray],
    metrics: Sequence[str] = REPORTED_METRICS,
    labels: tuple[str, str] = ("base", "hybrid"),
) -> dict[str, dict[str, float]]:
    """Candidate minus reference per metric: seed means and a paired percentile bootstrap.

    Seed i of the candidate is paired with seed i of the reference, both scored
    on the same TEST resamples; each resample contributes the mean over seeds of
    the paired differences.
    """
    rows = {}
    for column, name in enumerate(metrics):
        reference = np.asarray([values[name] for values in reference_point])
        candidate = np.asarray([values[name] for values in candidate_point])
        differences = candidate - reference
        bootstrapped = [
            float(
                np.mean(
                    [
                        candidate_values[row, column] - reference_values[row, column]
                        for reference_values, candidate_values in zip(
                            reference_resampled, candidate_resampled
                        )
                    ]
                )
            )
            for row in range(len(reference_resampled[0]))
        ]
        low, high = np.quantile(bootstrapped, [0.025, 0.975])
        rows[name] = {
            f"{labels[0]}_mean_over_seeds": float(np.mean(reference)),
            f"{labels[0]}_seed_std": _seed_std(reference),
            f"{labels[1]}_mean_over_seeds": float(np.mean(candidate)),
            f"{labels[1]}_seed_std": _seed_std(candidate),
            "difference_mean_over_seeds": float(np.mean(differences)),
            "difference_seed_std": _seed_std(differences),
            "paired_bootstrap_ci_low": float(low),
            "paired_bootstrap_ci_high": float(high),
        }
    return rows


def seed_mean_interval(
    point: Sequence[dict[str, float]],
    resampled: Sequence[np.ndarray],
    metrics: Sequence[str] = REPORTED_METRICS,
) -> dict[str, dict[str, float]]:
    """Mean over seeds, seed spread, and a percentile bootstrap of the seed-mean per metric."""
    summary = {}
    for column, name in enumerate(metrics):
        values = np.asarray([scores[name] for scores in point])
        means = [
            float(np.mean([seed_values[row, column] for seed_values in resampled]))
            for row in range(len(resampled[0]))
        ]
        low, high = np.quantile(means, [0.025, 0.975])
        summary[name] = {
            "mean": float(np.mean(values)),
            "seed_std": _seed_std(values),
            "bootstrap95_low": float(low),
            "bootstrap95_high": float(high),
        }
    return summary


def spearman_brown(rho: float, factor: float = 2.0) -> float:
    """Reliability of a measurement `factor` times as long as one with reliability rho."""
    return float(factor * rho / (1.0 + (factor - 1.0) * rho))


def split_half(repeats: Sequence[float], interleaved: bool = False) -> tuple[float, float]:
    """Medians of two halves of one schedule's repeats.

    By default the first len//2 repeats against the rest; with interleaved, the
    odd- against the even-numbered repeats, which a trend along the repeat
    sequence (such as warm-up) does not separate.
    """
    times = np.asarray(repeats, dtype=float)
    if times.size < 2:
        raise ValueError("a split half needs at least two repeats")
    if interleaved:
        return float(np.median(times[0::2])), float(np.median(times[1::2]))
    midpoint = times.size // 2
    return float(np.median(times[:midpoint])), float(np.median(times[midpoint:]))


def value_distribution(values: Sequence[float], max_listed: int = 12) -> dict[str, Any]:
    """Quantiles, the share equal to 1 (no penalty), and the distinct values when few."""
    values = np.asarray(values, dtype=float)
    distinct, counts = np.unique(values, return_counts=True)
    listed = (
        "; ".join(f"{value:g}:{count}" for value, count in zip(distinct, counts))
        if distinct.size <= max_listed
        else f"{distinct.size} distinct values"
    )
    return {
        "n": int(values.size),
        "n_unique": int(distinct.size),
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
        "fraction_equal_1": float(np.mean(values == 1.0)),
        "distinct_values": listed,
    }


def discordant_pairs(first: Sequence[float], second: Sequence[float]) -> tuple[int, int]:
    """Pairs ordered strictly oppositely by the two scores, and the number of pairs."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("scores must be one-dimensional arrays of equal size")
    upper = np.triu_indices(first.size, k=1)
    sign_first = np.sign(first[:, None] - first[None, :])[upper]
    sign_second = np.sign(second[:, None] - second[None, :])[upper]
    return int(np.sum(sign_first * sign_second < 0)), int(upper[0].size)
