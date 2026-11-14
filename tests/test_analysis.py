import numpy as np
import pytest

from dvcm.analysis import (
    REPORTED_METRICS,
    bootstrap_samples,
    discordant_pairs,
    paired_difference,
    resampled_metrics,
    resampled_metrics_parallel,
    seed_mean_interval,
    spearman_brown,
    split_half,
    value_distribution,
)
from dvcm.metrics import ranking_metrics


def _data(n=60, seeds=3, seed=7):
    rng = np.random.default_rng(seed)
    truth = rng.lognormal(3.0, 0.8, n)
    base = [truth * rng.lognormal(0.0, 0.4, n) for _ in range(seeds)]
    hybrid = [truth * rng.lognormal(0.0, 0.3, n) for _ in range(seeds)]
    return truth, base, hybrid


def _old_paired_loop(truth, base, hybrid, rng_seed, n_resamples):
    """The paired loop of scripts/05_evaluate.py before this module existed."""
    metric_names = REPORTED_METRICS
    base_values = [ranking_metrics(truth, prediction) for prediction in base]
    hybrid_values = [ranking_metrics(truth, prediction) for prediction in hybrid]
    rng = np.random.default_rng(rng_seed)
    bootstrapped = {name: [] for name in metric_names}
    for _ in range(n_resamples):
        sample = rng.integers(0, len(truth), size=len(truth))
        deltas = {name: [] for name in metric_names}
        for base_prediction, hybrid_prediction in zip(base, hybrid):
            base_metrics = ranking_metrics(truth[sample], base_prediction[sample])
            hybrid_metrics = ranking_metrics(truth[sample], hybrid_prediction[sample])
            for name in metric_names:
                deltas[name].append(hybrid_metrics[name] - base_metrics[name])
        for name in metric_names:
            bootstrapped[name].append(float(np.mean(deltas[name])))
    rows = {}
    for name in metric_names:
        base_seed_values = np.asarray([values[name] for values in base_values])
        hybrid_seed_values = np.asarray([values[name] for values in hybrid_values])
        differences = hybrid_seed_values - base_seed_values
        low, high = np.quantile(bootstrapped[name], [0.025, 0.975])
        rows[name] = {
            "base_mean_over_seeds": float(np.mean(base_seed_values)),
            "base_seed_std": float(np.std(base_seed_values, ddof=1)),
            "hybrid_mean_over_seeds": float(np.mean(hybrid_seed_values)),
            "hybrid_seed_std": float(np.std(hybrid_seed_values, ddof=1)),
            "difference_mean_over_seeds": float(np.mean(differences)),
            "difference_seed_std": float(np.std(differences, ddof=1)),
            "paired_bootstrap_ci_low": float(low),
            "paired_bootstrap_ci_high": float(high),
        }
    return rows


def test_bootstrap_samples_are_the_sequential_draws_of_one_generator():
    rng = np.random.default_rng(11)
    expected = [rng.integers(0, 9, size=9) for _ in range(4)]
    for got, want in zip(bootstrap_samples(9, 4, 11), expected):
        assert np.array_equal(got, want)


def test_resampled_metrics_score_each_bootstrap_sample():
    truth, base, _ = _data()
    values = resampled_metrics(truth, base[0], seed=3, n_resamples=5)
    assert values.shape == (5, len(REPORTED_METRICS))
    for row, sample in zip(values, bootstrap_samples(len(truth), 5, 3)):
        scores = ranking_metrics(truth[sample], base[0][sample])
        assert row.tolist() == [scores[name] for name in REPORTED_METRICS]


def test_parallel_resampled_metrics_equal_the_serial_values():
    truth, base, hybrid = _data()
    tasks = [(truth, prediction, 5) for prediction in base + hybrid]
    parallel = resampled_metrics_parallel(tasks, n_resamples=20, workers=2)
    for (y, prediction, seed), values in zip(tasks, parallel):
        assert np.array_equal(values, resampled_metrics(y, prediction, seed, 20))


def test_paired_difference_reproduces_the_previous_paired_comparison_exactly():
    truth, base, hybrid = _data()
    expected = _old_paired_loop(truth, base, hybrid, rng_seed=20260, n_resamples=50)
    base_point = [ranking_metrics(truth, prediction) for prediction in base]
    hybrid_point = [ranking_metrics(truth, prediction) for prediction in hybrid]
    base_resampled = [resampled_metrics(truth, p, 20260, 50) for p in base]
    hybrid_resampled = [resampled_metrics(truth, p, 20260, 50) for p in hybrid]

    rows = paired_difference(base_point, hybrid_point, base_resampled, hybrid_resampled)

    assert list(rows) == list(REPORTED_METRICS)
    for name in REPORTED_METRICS:
        assert rows[name] == expected[name]
        assert list(rows[name]) == list(expected[name])  # CSV column order


def test_paired_difference_labels_name_the_compared_scorers():
    truth, base, hybrid = _data(seeds=2)
    point = [ranking_metrics(truth, p) for p in base]
    resampled = [resampled_metrics(truth, p, 1, 10) for p in base]
    rows = paired_difference(point, point, resampled, resampled, labels=("nano_only", "adapted"))
    assert rows["spearman"]["difference_mean_over_seeds"] == 0.0
    assert rows["spearman"]["paired_bootstrap_ci_low"] == rows["spearman"]["paired_bootstrap_ci_high"] == 0.0
    assert "nano_only_mean_over_seeds" in rows["spearman"] and "adapted_seed_std" in rows["spearman"]


def test_seed_mean_interval_averages_seeds_inside_each_resample():
    truth, base, _ = _data(seeds=2)
    point = [ranking_metrics(truth, p) for p in base]
    resampled = [resampled_metrics(truth, p, 2, 40) for p in base]
    summary = seed_mean_interval(point, resampled)
    column = REPORTED_METRICS.index("spearman")
    means = [float(np.mean([values[r, column] for values in resampled])) for r in range(40)]
    assert summary["spearman"]["mean"] == float(np.mean([p["spearman"] for p in point]))
    assert summary["spearman"]["bootstrap95_low"] == float(np.quantile(means, 0.025))
    assert summary["spearman"]["bootstrap95_high"] == float(np.quantile(means, 0.975))
    single = seed_mean_interval(point[:1], resampled[:1])
    assert np.isnan(single["spearman"]["seed_std"])


def test_spearman_brown_doubles_the_length_of_a_split_half():
    assert spearman_brown(0.5) == pytest.approx(2 / 3)
    assert spearman_brown(1.0) == 1.0
    assert spearman_brown(0.0) == 0.0
    assert np.isnan(spearman_brown(float("nan")))


def test_split_half_uses_the_first_half_and_the_remaining_repeats():
    first, second = split_half([1.0, 3.0, 10.0, 20.0, 30.0])  # repeats 1-2 vs 3-5
    assert (first, second) == (2.0, 20.0)
    first, second = split_half(list(range(15)))  # repeats 1-7 vs 8-15
    assert (first, second) == (3.0, 10.5)
    with pytest.raises(ValueError):
        split_half([1.0])


def test_value_distribution_reports_unpenalised_share_and_distinct_values():
    summary = value_distribution([1.0, 1.0, 0.5, 0.25])
    assert summary["n"] == 4 and summary["n_unique"] == 3
    assert summary["fraction_equal_1"] == 0.5
    assert summary["min"] == 0.25 and summary["max"] == 1.0 and summary["median"] == 0.75
    assert summary["distinct_values"] == "0.25:1; 0.5:1; 1:2"
    wide = value_distribution(np.linspace(0.0, 1.0, 50))
    assert wide["distinct_values"] == "50 distinct values"


def test_discordant_pairs_count_pairs_ordered_differently():
    assert discordant_pairs([1, 2, 3], [1, 3, 2]) == (1, 3)
    assert discordant_pairs([1, 2, 3], [1, 2, 3]) == (0, 3)
    assert discordant_pairs([1, 2, 3, 4], [4, 3, 2, 1]) == (6, 6)
    assert discordant_pairs([1, 1, 2], [2, 1, 3]) == (0, 3)  # ties are not discordant
