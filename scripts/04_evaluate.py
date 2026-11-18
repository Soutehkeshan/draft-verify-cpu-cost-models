
#!/usr/bin/env python3
"""Evaluate all rankers on the held-out test split only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.analysis import (
    REPORTED_METRICS,
    discordant_pairs,
    paired_difference,
    resampled_metrics_parallel,
    seed_mean_interval,
    spearman_brown,
    split_half,
    value_distribution,
)
from dvcm.analytic import (
    AnalyticConstants,
    Penalties,
    constants_from_payload,
    latency_components,
    latency_seconds,
)
from dvcm.config import (
    add_target_argument,
    experiment_metadata,
    load_target_config,
    load_workload_config,
)
from dvcm.features import features
from dvcm.metrics import (
    ranking_metrics,
    spearman,
)
from dvcm.provenance import capture_provenance, write_csv
from dvcm.study import (
    load_layouts,
    load_pool,
    load_splits,
    validate_splits,
)


PAIRED_RESAMPLES = 2000
TRAINING_SEEDS = tuple(range(5))
LEARNING_MODELS = (("base", "xgb(base)"), ("base_analytic", "xgb(base+analytic)"))


def _read_constants(
    results_dir: Path,
) -> tuple[AnalyticConstants, AnalyticConstants, dict[str, AnalyticConstants]]:
    with (results_dir / "constants.json").open(encoding="utf-8") as source:
        hand_set, fitted, nested = constants_from_payload(json.load(source))
    if nested.get("all") != fitted:
        raise ValueError("constants.json: the 'all' nested constants differ from the TRAIN fit")
    return hand_set, fitted, nested


def _model_features(frame, layouts, include_derived, constants):
    matrices = []
    for row in frame.itertuples(index=False):
        position = np.asarray(row.vector, dtype=float)
        layout = layouts[int(row.sketch_id)]
        values = features(position, int(row.sketch_id), layout, include_derived)
        if include_derived:
            values = np.append(values, latency_seconds(position, layout, constants) * 1000)
        matrices.append(values)
    return np.asarray(matrices, dtype=float)


def _predict_model_seed(
    frame,
    layouts,
    constants,
    feature_set,
    size_name,
    seed,
    results_dir,
):
    include_derived = feature_set == "base_analytic"
    values = _model_features(frame, layouts, include_derived, constants)
    model = XGBRegressor()
    model.load_model(
        results_dir / "models" / f"{feature_set}_n{size_name}_seed{seed}.json"
    )
    return np.exp(model.predict(values))


def _metric_values(y_true, prediction):
    return ranking_metrics(y_true, prediction)


def _bootstrap_metrics(y_true, prediction, seed=0, n_resamples=1000):
    rng = np.random.default_rng(seed)
    values = {name: [] for name in _metric_values(y_true, prediction)}
    for _ in range(n_resamples):
        indexes = rng.integers(0, len(y_true), size=len(y_true))
        observed = _metric_values(y_true[indexes], prediction[indexes])
        for name, value in observed.items():
            if np.isfinite(value):
                values[name].append(float(value))
    return {
        name: np.quantile(observed, [0.025, 0.975]).tolist()
        if observed
        else [float("nan"), float("nan")]
        for name, observed in values.items()
    }


def _random_metrics(y_true, n_draws=1000, seed=0):
    rng = np.random.default_rng(seed)
    draws = {
        name: []
        for name in (
            "spearman",
            "fastest_decile_spearman",
            "top_10_recall_in_30",
            *(f"top_{k}_latency_ratio" for k in (1, 5, 10)),
        )
    }
    for _ in range(n_draws):
        measured = _metric_values(y_true, rng.random(len(y_true)))
        for name, value in measured.items():
            if np.isfinite(value):
                draws[name].append(float(value))
    summary = {}
    for name, values in draws.items():
        summary[name] = (
            float(np.mean(values)),
            np.quantile(values, [0.025, 0.975]).tolist(),
        )
    return summary


def _score_row(scope, name, y_true, prediction, seed=0, random_summary=None):
    if random_summary is None:
        values = _metric_values(y_true, prediction)
        intervals = _bootstrap_metrics(y_true, prediction, seed=seed)
    else:
        values = {
            metric: estimate
            for metric, (estimate, _) in random_summary.items()
        }
        intervals = {
            metric: bounds for metric, (_, bounds) in random_summary.items()
        }
    metrics = {
        "scope": scope,
        "scorer": name,
        "n_test": len(y_true),
        "fastest_decile_n": max(2, int(np.ceil(len(y_true) * 0.1))),
        "top_10_recall_true_k": min(10, len(y_true)),
        "top_10_recall_predicted_k": min(30, len(y_true)),
        "top_k_ratio_definition": "mean selected latency / fastest test latency; lower is better",
        "spearman": values["spearman"],
        "spearman_ci_low": intervals["spearman"][0],
        "spearman_ci_high": intervals["spearman"][1],
        "fastest_decile_spearman": values["fastest_decile_spearman"],
        "fastest_decile_spearman_ci_low": intervals["fastest_decile_spearman"][0],
        "fastest_decile_spearman_ci_high": intervals["fastest_decile_spearman"][1],
        "top_10_recall_in_30": values["top_10_recall_in_30"],
        "top_10_recall_ci_low": intervals["top_10_recall_in_30"][0],
        "top_10_recall_ci_high": intervals["top_10_recall_in_30"][1],
    }
    for k in (1, 5, 10):
        metrics[f"top_{k}_latency_ratio"] = values[f"top_{k}_latency_ratio"]
        metrics[f"top_{k}_ci_low"] = intervals[f"top_{k}_latency_ratio"][0]
        metrics[f"top_{k}_ci_high"] = intervals[f"top_{k}_latency_ratio"][1]
    return metrics


def _scope_indices(frame):
    yield "pooled", np.arange(len(frame), dtype=int)
    for sketch_id in sorted(int(value) for value in frame["sketch_id"].unique()):
        yield (
            f"sketch_{sketch_id}",
            np.flatnonzero(frame["sketch_id"].to_numpy(dtype=int) == sketch_id),
        )


def _draft_verify_ratio(y_true, draft_scores, verify_scores, k):
    drafted = np.argsort(draft_scores, kind="stable")[: min(k, len(y_true))]
    selected = int(drafted[np.argmin(verify_scores[drafted])])
    return float(y_true[selected] / np.min(y_true))


def _draft_only_ratio(y_true, draft_scores, k):
    drafted = np.argsort(draft_scores, kind="stable")[: min(k, len(y_true))]
    selected = int(drafted[np.argmin(draft_scores[drafted])])
    return float(y_true[selected] / np.min(y_true))


def _write_paired_and_learning_curve(
    test, layouts, fitted, nested, splits, y_true, scopes, provenance, results_dir
):
    """Paired base-vs-hybrid comparison at every nested TRAIN size, on shared TEST resamples.

    Each scope draws its resamples from seed 20260 + scope index, so the
    full-TRAIN rows are the paired comparison of paired_comparison.csv.
    """
    sizes = list(splits["nested_train"])
    train_sizes = {name: len(ids) for name, ids in splits["nested_train"].items()}
    predictions = {
        (feature_set, size_name, seed): _predict_model_seed(
            test,
            layouts,
            nested[size_name] if feature_set == "base_analytic" else fitted,
            feature_set,
            size_name,
            seed,
            results_dir,
        )
        for feature_set, _ in LEARNING_MODELS
        for size_name in sizes
        for seed in TRAINING_SEEDS
    }
    keys = list(predictions)
    tasks = [
        (y_true[indexes], predictions[key][indexes], 20260 + scope_index)
        for scope_index, (_, indexes) in enumerate(scopes)
        for key in keys
    ]
    resampled_values = iter(resampled_metrics_parallel(tasks, PAIRED_RESAMPLES))
    resampled = {
        (scope, key): next(resampled_values) for scope, _ in scopes for key in keys
    }

    seed_rows, summary_rows, paired_rows, comparison_rows = [], [], [], []
    for scope, indexes in scopes:
        truth = y_true[indexes]
        point = {
            key: _metric_values(truth, prediction[indexes])
            for key, prediction in predictions.items()
        }
        for feature_set, model in LEARNING_MODELS:
            for size_name in sizes:
                model_keys = [(feature_set, size_name, seed) for seed in TRAINING_SEEDS]
                constants_note = (
                    "fitted on this subset" if feature_set == "base_analytic" else "not used"
                )
                for key in model_keys:
                    seed_rows.append(
                        {
                            "scope": scope,
                            "model": model,
                            "feature_set": feature_set,
                            "train_size": train_sizes[size_name],
                            "subset": size_name,
                            "seed": key[2],
                            "analytic_constants": constants_note,
                            **{name: point[key][name] for name in REPORTED_METRICS},
                        }
                    )
                summary = seed_mean_interval(
                    [point[key] for key in model_keys],
                    [resampled[(scope, key)] for key in model_keys],
                )
                summary_rows.append(
                    {
                        "scope": scope,
                        "model": model,
                        "feature_set": feature_set,
                        "train_size": train_sizes[size_name],
                        "subset": size_name,
                        "n_test": len(indexes),
                        "seed_count": len(model_keys),
                        "analytic_constants": constants_note,
                        **{
                            f"{name}_{statistic}": value
                            for name, statistics in summary.items()
                            for statistic, value in statistics.items()
                        },
                        "bootstrap_resamples": PAIRED_RESAMPLES,
                    }
                )
        for size_name in sizes:
            base_keys = [("base", size_name, seed) for seed in TRAINING_SEEDS]
            hybrid_keys = [("base_analytic", size_name, seed) for seed in TRAINING_SEEDS]
            differences = paired_difference(
                [point[key] for key in base_keys],
                [point[key] for key in hybrid_keys],
                [resampled[(scope, key)] for key in base_keys],
                [resampled[(scope, key)] for key in hybrid_keys],
            )
            for name in REPORTED_METRICS:
                row = {
                    "scope": scope,
                    "n_test": len(indexes),
                    "metric": name,
                    "difference_definition": "xgb(base+analytic) minus xgb(base)",
                    **differences[name],
                    "bootstrap_resamples": PAIRED_RESAMPLES,
                    "training_seeds": len(TRAINING_SEEDS),
                }
                paired_rows.append(
                    {"train_size": train_sizes[size_name], "subset": size_name, **row}
                )
                if size_name == "all":
                    comparison_rows.append(row)
    write_csv(pd.DataFrame(comparison_rows), results_dir / "paired_comparison.csv", provenance)
    write_csv(pd.DataFrame(seed_rows), results_dir / "learning_curve.csv", provenance)
    write_csv(
        pd.DataFrame(summary_rows), results_dir / "learning_curve_summary.csv", provenance
    )
    write_csv(
        pd.DataFrame(paired_rows), results_dir / "learning_curve_paired.csv", provenance
    )


def _scorer_rows(scopes, y_true, predictions):
    """evaluation.csv rows: random ranking, then every scorer, per scope."""
    score_rows = []
    random_summaries = {}
    for scope_index, (scope, indexes) in enumerate(scopes):
        scoped_truth = y_true[indexes]
        random_summary = _random_metrics(
            scoped_truth,
            n_draws=1000,
            seed=10_000 + scope_index,
        )
        random_summaries[scope] = random_summary
        score_rows.append(
            _score_row(
                scope,
                "random",
                scoped_truth,
                np.zeros(len(indexes)),
                seed=scope_index,
                random_summary=random_summary,
            )
        )
        for scorer_index, (name, prediction) in enumerate(predictions.items()):
            score_rows.append(
                _score_row(
                    scope,
                    name,
                    scoped_truth,
                    prediction[indexes],
                    seed=scope_index * 100 + scorer_index,
                )
            )
    return score_rows, random_summaries


def _ceiling(reliability):
    return float(np.sqrt(reliability)) if np.isfinite(reliability) and reliability >= 0 else float("nan")


def _noise_row(scope, frame, median_column, repeats_column):
    """Repeat-half agreement of one timing table and its Spearman-Brown full-length estimate.

    The ordered split compares the first half of each schedule's repeats with
    the rest; the interleaved split compares odd- with even-numbered repeats,
    which a trend along the repeat sequence (such as warm-up) does not separate.
    The fastest decile is the fastest 10% by the table's own median.
    """
    pairs = []
    repeat_counts = Counter()
    for median, repeats in zip(frame[median_column], frame[repeats_column]):
        times = np.asarray(repeats, dtype=float)
        if times.size >= 2:
            pairs.append(
                (float(median), *split_half(times), *split_half(times, interleaved=True))
            )
            repeat_counts[times.size] += 1
    pairs.sort(key=lambda item: item[0])
    halves = np.asarray([item[1:] for item in pairs], dtype=float).reshape(-1, 4)
    decile_n = max(2, int(np.ceil(len(pairs) * 0.1))) if len(pairs) >= 2 else len(pairs)

    def agreement(first, second):
        return float(spearmanr(first, second).statistic) if len(first) > 1 else float("nan")

    whole = agreement(halves[:, 0], halves[:, 1])
    decile = agreement(halves[:decile_n, 0], halves[:decile_n, 1])
    interleaved_whole = agreement(halves[:, 2], halves[:, 3])
    interleaved_decile = agreement(halves[:decile_n, 2], halves[:decile_n, 3])
    repeats = repeat_counts.most_common(1)[0][0] if repeat_counts else 0
    half = repeats // 2
    ordered_ratio = halves[:, 0] / halves[:, 1]
    row = {
        "scope": scope,
        "repeat_half_spearman": whole,
        "n_repeat_pairs": len(pairs),
        "top_decile_n": decile_n,
        "top_decile_repeat_half_spearman": decile,
        "repeat_split": (
            f"median of repeats 1-{half} vs {half + 1}-{repeats}" if repeats else "n/a"
        ),
        "repeat_half_spearman_brown": spearman_brown(whole),
        "top_decile_repeat_half_spearman_brown": spearman_brown(decile),
    }
    row["noise_ceiling_sqrt_spearman_brown"] = _ceiling(row["repeat_half_spearman_brown"])
    row["top_decile_noise_ceiling_sqrt_spearman_brown"] = _ceiling(
        row["top_decile_repeat_half_spearman_brown"]
    )
    row["interleaved_split"] = "median of odd- vs even-numbered repeats" if repeats else "n/a"
    for prefix, rho in (("", interleaved_whole), ("top_decile_", interleaved_decile)):
        row[f"{prefix}interleaved_repeat_half_spearman"] = rho
        row[f"{prefix}interleaved_repeat_half_spearman_brown"] = spearman_brown(rho)
        row[f"{prefix}interleaved_noise_ceiling_sqrt_spearman_brown"] = _ceiling(
            spearman_brown(rho)
        )
    row["first_half_over_rest_median_ratio"] = (
        float(np.median(ordered_ratio)) if len(pairs) else float("nan")
    )
    row["share_first_half_slower"] = (
        float(np.mean(ordered_ratio > 1.0)) if len(pairs) else float("nan")
    )
    return row


def _write_analytic_diagnosis(
    test, layouts, hand_set, fitted, y_true, scopes, provenance, results_dir
):
    """Compute/memory shares, penalty distributions and the cache penalties' ranking effect."""
    constant_sets = (("fitted", fitted), ("hand_set", hand_set))
    positions = [np.asarray(row.vector, dtype=float) for row in test.itertuples(index=False)]
    sketch_layouts = [layouts[int(value)] for value in test["sketch_id"]]
    parts = {
        name: [
            latency_components(position, layout, constants)
            for position, layout in zip(positions, sketch_layouts)
        ]
        for name, constants in constant_sets
    }
    without_cache = {
        name: np.asarray(
            [
                latency_seconds(position, layout, constants, Penalties(cache=False))
                for position, layout in zip(positions, sketch_layouts)
            ]
        )
        for name, constants in constant_sets
    }
    share_rows, penalty_rows, cache_rows = [], [], []
    for scope, indexes in scopes:
        truth = y_true[indexes]
        for name, _ in constant_sets:
            scoped = [parts[name][index] for index in indexes]
            compute = np.asarray([item["compute_time_s"] for item in scoped])
            memory = np.asarray([item["memory_time_s"] for item in scoped])
            share = compute / (compute + memory)
            share_rows.append(
                {
                    "scope": scope,
                    "constants": name,
                    "n_test": len(indexes),
                    "compute_share_median": float(np.median(share)),
                    "compute_share_p10": float(np.quantile(share, 0.10)),
                    "compute_share_p90": float(np.quantile(share, 0.90)),
                    "memory_share_median": float(np.median(1.0 - share)),
                    "memory_share_p10": float(np.quantile(1.0 - share, 0.10)),
                    "memory_share_p90": float(np.quantile(1.0 - share, 0.90)),
                    "memory_dominated_fraction": float(np.mean(share < 0.5)),
                    "compute_time_ms_median": float(np.median(compute) * 1000),
                    "memory_time_ms_median": float(np.median(memory) * 1000),
                    "memory_time_ms_max": float(np.max(memory) * 1000),
                    "share_definition": "compute_time / (compute_time + memory_time) per TEST schedule",
                }
            )
            full = compute + memory
            reduced = without_cache[name][indexes]
            discordant, pair_count = discordant_pairs(full, reduced)
            increase = (full / reduced - 1.0) * 100.0
            cache_l1 = np.asarray([item["penalty_cache_l1"] for item in scoped])
            cache_l2 = np.asarray([item["penalty_cache_l2"] for item in scoped])
            with_metrics = _metric_values(truth, full)
            without_metrics = _metric_values(truth, reduced)
            cache_rows.append(
                {
                    "scope": scope,
                    "constants": name,
                    "n_test": len(indexes),
                    "schedules_with_l1_cache_penalty": int(np.sum(cache_l1 < 1.0)),
                    "schedules_with_l2_cache_penalty": int(np.sum(cache_l2 < 1.0)),
                    "predicted_latency_increase_pct_median": float(np.median(increase)),
                    "predicted_latency_increase_pct_max": float(np.max(increase)),
                    "spearman_with_vs_without_cache_penalty": spearman(full, reduced),
                    "discordant_pairs": discordant,
                    "pairs": pair_count,
                    "discordant_pair_fraction": discordant / pair_count if pair_count else float("nan"),
                    "same_predicted_top_1": bool(np.argmin(full) == np.argmin(reduced)),
                    "predicted_top_10_overlap": len(
                        set(np.argsort(full, kind="stable")[:10].tolist())
                        & set(np.argsort(reduced, kind="stable")[:10].tolist())
                    ),
                    "measured_spearman_with_cache_penalty": with_metrics["spearman"],
                    "measured_spearman_without_cache_penalty": without_metrics["spearman"],
                    "measured_top_1_ratio_with_cache_penalty": with_metrics["top_1_latency_ratio"],
                    "measured_top_1_ratio_without_cache_penalty": without_metrics["top_1_latency_ratio"],
                    "cache_penalties_change_ranking": bool(discordant > 0),
                }
            )
        scoped = [parts["fitted"][index] for index in indexes]
        for factor, applies_to in (
            ("penalty_vector", "compute_time"),
            ("penalty_parallel", "compute_time"),
            ("penalty_register", "compute_time"),
            ("compute_penalty", "compute_time"),
            ("penalty_cache_l1", "memory_time"),
            ("penalty_cache_l2", "memory_time"),
            ("memory_penalty", "memory_time"),
        ):
            penalty_rows.append(
                {
                    "scope": scope,
                    "penalty": factor,
                    "divides": applies_to,
                    **value_distribution([item[factor] for item in scoped]),
                    "note": "1.0 means no penalty; identical for hand-set and fitted constants",
                }
            )
    write_csv(pd.DataFrame(share_rows), results_dir / "analytic_time_shares.csv", provenance)
    write_csv(pd.DataFrame(penalty_rows), results_dir / "analytic_penalties.csv", provenance)
    write_csv(pd.DataFrame(cache_rows), results_dir / "analytic_cache_effect.csv", provenance)


def _write_retime_outputs(test, data_dir, predictions, provenance, results_dir):
    """Compare the collection medians with a second timing of TEST, when one exists."""
    path = data_dir / "test_retime.parquet"
    if not path.exists():
        return None
    meta = json.loads((data_dir / "test_retime_meta.json").read_text(encoding="utf-8"))
    if meta.get("state") != "complete":
        raise ValueError("test_retime_meta.json does not describe a complete re-timing")
    retimed = pd.read_parquet(path).set_index("id")
    ids = test["id"].astype(int).to_numpy()
    if sorted(retimed.index.astype(int)) != sorted(ids.tolist()):
        raise ValueError("test_retime.parquet ids differ from the TEST ids")
    retimed = retimed.loc[ids]
    for (sketch_id, vector), (re_sketch, re_vector) in zip(
        zip(test["sketch_id"], test["vector"]), zip(retimed["sketch_id"], retimed["vector"])
    ):
        if int(sketch_id) != int(re_sketch) or list(map(int, vector)) != list(map(int, re_vector)):
            raise ValueError("test_retime.parquet schedules differ from the pool's TEST schedules")
    keep = np.flatnonzero(
        retimed["status"].isin(("ok", "slow_skip")).to_numpy()
        & np.isfinite(retimed["median_ms"].to_numpy(dtype=float))
        & (retimed["median_ms"].to_numpy(dtype=float) > 0)
    )
    original = test.iloc[keep].reset_index(drop=True)
    second = retimed.iloc[keep].reset_index()
    scopes = list(_scope_indices(original))
    original_truth = original["median_ms"].to_numpy(dtype=float)
    retimed_truth = second["median_ms"].to_numpy(dtype=float)
    scoped_predictions = {name: values[keep] for name, values in predictions.items()}
    source = {
        "retime_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "retime_git_commit": meta["provenance"]["git_commit"],
        "original_repeats": int(original["repeats_ms"].map(len).max()),
        "retimed_repeats": int(meta["repeats"]),
    }

    agreement_rows = []
    for scope, indexes in scopes:
        first, again = original_truth[indexes], retimed_truth[indexes]
        decile_n = max(2, int(np.ceil(len(indexes) * 0.1)))
        fastest = np.argsort(first, kind="stable")[:decile_n]
        first_noise = _noise_row(scope, original.iloc[indexes], "median_ms", "repeats_ms")
        again_noise = _noise_row(scope, second.iloc[indexes], "median_ms", "repeats_ms")
        ratio = again / first
        agreement_rows.append(
            {
                "scope": scope,
                "n_schedules": len(indexes),
                "spearman_original_vs_retimed": spearman(first, again),
                "fastest_decile_n": decile_n,
                "fastest_decile_spearman_original_vs_retimed": spearman(
                    first[fastest], again[fastest]
                ),
                "fastest_decile_definition": "fastest 10% by the original median",
                "original_repeat_half_spearman_brown": first_noise["repeat_half_spearman_brown"],
                "retimed_repeat_split": again_noise["repeat_split"],
                "retimed_repeat_half_spearman": again_noise["repeat_half_spearman"],
                "retimed_repeat_half_spearman_brown": again_noise["repeat_half_spearman_brown"],
                "expected_spearman_if_only_repeat_noise": _ceiling(
                    first_noise["repeat_half_spearman_brown"]
                    * again_noise["repeat_half_spearman_brown"]
                ),
                "original_top_decile_repeat_half_spearman_brown": first_noise[
                    "top_decile_repeat_half_spearman_brown"
                ],
                "retimed_top_decile_repeat_half_spearman": again_noise[
                    "top_decile_repeat_half_spearman"
                ],
                "retimed_top_decile_repeat_half_spearman_brown": again_noise[
                    "top_decile_repeat_half_spearman_brown"
                ],
                "expected_top_decile_spearman_if_only_repeat_noise": _ceiling(
                    first_noise["top_decile_repeat_half_spearman_brown"]
                    * again_noise["top_decile_repeat_half_spearman_brown"]
                ),
                **{
                    f"{prefix}{name}": value
                    for prefix, name, value in (
                        ("original_", "interleaved_repeat_half_spearman_brown",
                         first_noise["interleaved_repeat_half_spearman_brown"]),
                        ("retimed_", "interleaved_repeat_half_spearman_brown",
                         again_noise["interleaved_repeat_half_spearman_brown"]),
                        ("", "expected_spearman_if_only_repeat_noise_interleaved", _ceiling(
                            first_noise["interleaved_repeat_half_spearman_brown"]
                            * again_noise["interleaved_repeat_half_spearman_brown"])),
                        ("original_", "top_decile_interleaved_repeat_half_spearman_brown",
                         first_noise["top_decile_interleaved_repeat_half_spearman_brown"]),
                        ("retimed_", "top_decile_interleaved_repeat_half_spearman_brown",
                         again_noise["top_decile_interleaved_repeat_half_spearman_brown"]),
                        ("", "expected_top_decile_spearman_if_only_repeat_noise_interleaved", _ceiling(
                            first_noise["top_decile_interleaved_repeat_half_spearman_brown"]
                            * again_noise["top_decile_interleaved_repeat_half_spearman_brown"])),
                    )
                },
                "median_ratio_retimed_over_original": float(np.median(ratio)),
                "ratio_p10": float(np.quantile(ratio, 0.10)),
                "ratio_p90": float(np.quantile(ratio, 0.90)),
                "fraction_differing_over_5pct": float(np.mean(np.abs(ratio - 1.0) > 0.05)),
                **source,
            }
        )
    write_csv(pd.DataFrame(agreement_rows), results_dir / "retime_agreement.csv", provenance)

    evaluation_rows = []
    for truth_name, truth in (("original", original_truth), ("retimed", retimed_truth)):
        rows, _ = _scorer_rows(scopes, truth, scoped_predictions)
        repeats = source["original_repeats"] if truth_name == "original" else source["retimed_repeats"]
        evaluation_rows.extend(
            {"truth": truth_name, "truth_repeats": repeats, **row} for row in rows
        )
    write_csv(pd.DataFrame(evaluation_rows), results_dir / "retime_evaluation.csv", provenance)
    return agreement_rows[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_target_argument(parser)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    target = load_target_config(args.target, ROOT)
    workload = load_workload_config(root=ROOT)
    data_dir, results_dir = target.paths(ROOT)
    provenance = capture_provenance(
        ROOT,
        allow_dirty=args.allow_dirty,
        run_context=experiment_metadata(target, workload, data_dir),
    )

    frame = load_pool(data_dir=data_dir)
    splits = load_splits(data_dir)
    validate_splits(splits, set(frame["id"].astype(int)))
    layouts = load_layouts(data_dir, target)
    test_ids = set(int(value) for value in splits["test"])
    test = frame[frame["id"].astype(int).isin(test_ids)].reset_index(drop=True)
    if test.empty:
        raise ValueError("The test split contains no modelable measurements")
    hand_set, fitted, nested = _read_constants(results_dir)
    y_true = test["median_ms"].to_numpy(dtype=float)

    base_seed_predictions = [
        _predict_model_seed(
            test, layouts, fitted, "base", "all", seed, results_dir
        )
        for seed in range(5)
    ]
    hybrid_seed_predictions = [
        _predict_model_seed(
            test, layouts, fitted, "base_analytic", "all", seed, results_dir
        )
        for seed in range(5)
    ]
    predictions = {
        "analytic(hand_set)": np.asarray(
            [
                latency_seconds(
                    np.asarray(row.vector), layouts[int(row.sketch_id)], hand_set
                )
                * 1000
                for row in test.itertuples(index=False)
            ]
        ),
        "analytic(fitted)": np.asarray(
            [
                latency_seconds(
                    np.asarray(row.vector), layouts[int(row.sketch_id)], fitted
                )
                * 1000
                for row in test.itertuples(index=False)
            ]
        ),
        "xgb(base)": np.mean(base_seed_predictions, axis=0),
        "xgb(base+analytic)": np.mean(hybrid_seed_predictions, axis=0),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    scopes = list(_scope_indices(test))
    score_rows, random_summaries = _scorer_rows(scopes, y_true, predictions)
    write_csv(
        pd.DataFrame(score_rows),
        results_dir / "evaluation.csv",
        provenance,
    )

    pooled_random_recall = random_summaries["pooled"]["top_10_recall_in_30"][0]
    expected_random_hits = 30 * 10 / len(y_true)
    measured_random_hits = pooled_random_recall * 10
    print(
        "Random top-10 recall sanity: expected hits in top 30="
        f"30*10/{len(y_true)}={expected_random_hits:.4f}; "
        f"observed mean hits={measured_random_hits:.4f} "
        f"(recall fraction={pooled_random_recall:.4f}, expected "
        f"{30 / len(y_true):.4f})."
    )
    _write_paired_and_learning_curve(
        test, layouts, fitted, nested, splits, y_true, scopes, provenance, results_dir
    )

    draft_scores = {
        "analytic_fitted": predictions["analytic(fitted)"],
        "analytic_hand_set": predictions["analytic(hand_set)"],
        "xgb_base": predictions["xgb(base)"],
    }
    verify_predictions = predictions["xgb(base+analytic)"]
    draft_rows = []
    for scope_index, (scope, indexes) in enumerate(scopes):
        scoped_truth = y_true[indexes]
        verify_score = verify_predictions[indexes]
        true_order = np.argsort(scoped_truth, kind="stable")
        true_top_10 = set(true_order[: min(10, len(indexes))].tolist())
        for requested_k in (32, 64, 128, 256):
            scoped_drafts = {
                name: scores[indexes] for name, scores in draft_scores.items()
            }
            scoped_drafts["random"] = np.random.default_rng(
                40_000 + scope_index * 100 + requested_k
            ).random(len(indexes))
            for draft_method, draft_score in scoped_drafts.items():
                draft_order = np.argsort(draft_score, kind="stable")
                drafted = set(draft_order[: min(requested_k, len(indexes))].tolist())
                recall_top_10 = len(true_top_10 & drafted)
                draft_only_ratio = _draft_only_ratio(
                    scoped_truth, draft_score, requested_k
                )
                ratio = _draft_verify_ratio(
                    scoped_truth, draft_score, verify_score, requested_k
                )
                draft_rows.append(
                    {
                        "scope": scope,
                        "n_test": len(indexes),
                        "draft_method": draft_method,
                        "verify_method": "xgb(base+analytic)",
                        "K": requested_k,
                        "candidates": min(requested_k, len(indexes)),
                        "draft_recall_at_K_top_10_count": recall_top_10,
                        "draft_recall_at_K_top_10_fraction": recall_top_10
                        / max(1, len(true_top_10)),
                        "draft_only_top_1_latency_ratio": draft_only_ratio,
                        "top_1_latency_ratio": ratio,
                    }
                )
    write_csv(
        pd.DataFrame(draft_rows),
        results_dir / "draft_verify.csv",
        provenance,
    )

    ablation_predictions = {
        "xgb_base_features": predictions["xgb(base)"],
        "xgb_base_plus_derived_and_analytic": predictions["xgb(base+analytic)"],
    }
    for removed in ("vector", "parallel", "cache", "register"):
        penalties = Penalties(
            **{
                name: name != removed
                for name in ("vector", "parallel", "cache", "register")
            }
        )
        ablation_predictions[f"without_{removed}"] = np.asarray(
            [
                latency_seconds(
                    np.asarray(row.vector),
                    layouts[int(row.sketch_id)],
                    fitted,
                    penalties,
                )
                for row in test.itertuples(index=False)
            ]
        )
    # The full analytic models reuse their evaluation.csv bootstrap seeds, so
    # their rows here equal their rows there.
    full_analytic = {
        name: list(predictions).index(name)
        for name in ("analytic(fitted)", "analytic(hand_set)")
    }
    ablation_rows = []
    for scope_index, (scope, indexes) in enumerate(scopes):
        for model_index, (name, prediction) in enumerate(ablation_predictions.items()):
            ablation_rows.append(
                _score_row(
                    scope,
                    name,
                    y_true[indexes],
                    prediction[indexes],
                    seed=500 + scope_index * 100 + model_index,
                )
            )
        for name, scorer_index in full_analytic.items():
            ablation_rows.append(
                _score_row(
                    scope,
                    name,
                    y_true[indexes],
                    predictions[name][indexes],
                    seed=scope_index * 100 + scorer_index,
                )
            )
    all_penalties = ("vector", "parallel", "cache", "register")
    for row in ablation_rows:
        name = row["scorer"]
        removed = name.removeprefix("without_") if name.startswith("without_") else None
        row["constants"] = (
            "hand_set" if name == "analytic(hand_set)"
            else "n/a" if name == "xgb_base_features"
            else "fitted"
        )
        row["penalties_enabled"] = (
            "n/a" if name == "xgb_base_features"
            else ", ".join(penalty for penalty in all_penalties if penalty != removed)
        )
    write_csv(
        pd.DataFrame(ablation_rows),
        results_dir / "ablations.csv",
        provenance,
    )
    _write_analytic_diagnosis(
        test, layouts, hand_set, fitted, y_true, scopes, provenance, results_dir
    )

    noise_rows = [
        _noise_row(scope, test.iloc[indexes], "median_ms", "repeats_ms")
        for scope, indexes in scopes
    ]
    write_csv(pd.DataFrame(noise_rows), results_dir / "noise_floor.csv", provenance)

    retime = _write_retime_outputs(test, data_dir, predictions, provenance, results_dir)

    print(
        f"Evaluated {len(test)} held-out schedules "
        f"({max(2, int(np.ceil(len(test) * 0.1)))} in the fastest decile); "
        f"repeat-half Spearman={noise_rows[0]['repeat_half_spearman']:.3f} "
        f"(Spearman-Brown {noise_rows[0]['repeat_half_spearman_brown']:.3f}), "
        f"fastest-decile repeat-half Spearman="
        f"{noise_rows[0]['top_decile_repeat_half_spearman']:.3f} "
        f"(Spearman-Brown {noise_rows[0]['top_decile_repeat_half_spearman_brown']:.3f})"
    )
    if retime is None:
        print("No complete test_retime.parquet: re-timing comparison skipped")
    else:
        print(
            f"Re-timed TEST ({retime['n_schedules']} schedules, {retime['retimed_repeats']} repeats): "
            f"Spearman original vs re-timed {retime['spearman_original_vs_retimed']:.3f}, "
            f"fastest decile {retime['fastest_decile_spearman_original_vs_retimed']:.3f}"
        )


if __name__ == "__main__":
    main()
