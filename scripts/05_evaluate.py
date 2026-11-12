
#!/usr/bin/env python3
"""Evaluate all rankers on the held-out test split only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.analytic import AnalyticConstants, Penalties, latency_seconds
from dvcm.features import features
from dvcm.learned import train_xgb
from dvcm.metrics import (
    fastest_decile_spearman,
    spearman,
    top_k_latency_ratio,
    top_k_recall,
)
from dvcm.provenance import capture_provenance, write_csv
from dvcm.study import (
    RESULTS_DIR,
    load_layouts,
    load_pool,
    load_splits,
    validate_splits,
)


def _read_constants() -> tuple[AnalyticConstants, AnalyticConstants]:
    with (RESULTS_DIR / "constants.json").open(encoding="utf-8") as source:
        payload = json.load(source)
    return (
        AnalyticConstants(**payload["hand_set_constants"]),
        AnalyticConstants(**payload["constants"]),
    )


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


def _predict_model_seed(frame, layouts, constants, feature_set, size_name, seed):
    include_derived = feature_set == "base_analytic"
    values = _model_features(frame, layouts, include_derived, constants)
    model = XGBRegressor()
    model.load_model(
        RESULTS_DIR / "models" / f"{feature_set}_n{size_name}_seed{seed}.json"
    )
    return np.exp(model.predict(values))


def _metric_values(y_true, prediction):
    ranked = np.argsort(prediction, kind="stable")
    values = {
        "spearman": spearman(y_true, prediction),
        "fastest_decile_spearman": fastest_decile_spearman(y_true, prediction),
        "top_10_recall_in_30": top_k_recall(y_true, prediction, 10, 30),
    }
    for k in (1, 5, 10):
        values[f"top_{k}_latency_ratio"] = top_k_latency_ratio(
            y_true, ranked, min(k, len(y_true))
        )
    return values


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


def _paired_comparison(
    y_true,
    base_seed_predictions,
    hybrid_seed_predictions,
    scopes,
    provenance,
):
    rows = []
    metric_names = (
        "spearman",
        "fastest_decile_spearman",
        "top_10_recall_in_30",
        "top_1_latency_ratio",
        "top_5_latency_ratio",
    )
    for scope_index, (scope, indexes) in enumerate(scopes):
        truth = y_true[indexes]
        base = [prediction[indexes] for prediction in base_seed_predictions]
        hybrid = [prediction[indexes] for prediction in hybrid_seed_predictions]
        base_values = [_metric_values(truth, prediction) for prediction in base]
        hybrid_values = [_metric_values(truth, prediction) for prediction in hybrid]
        rng = np.random.default_rng(20260 + scope_index)
        bootstrapped = {name: [] for name in metric_names}
        for _ in range(2000):
            sample = rng.integers(0, len(truth), size=len(truth))
            deltas = {name: [] for name in metric_names}
            for base_prediction, hybrid_prediction in zip(base, hybrid):
                base_metrics = _metric_values(
                    truth[sample], base_prediction[sample]
                )
                hybrid_metrics = _metric_values(
                    truth[sample], hybrid_prediction[sample]
                )
                for name in metric_names:
                    deltas[name].append(hybrid_metrics[name] - base_metrics[name])
            for name in metric_names:
                bootstrapped[name].append(float(np.mean(deltas[name])))

        for name in metric_names:
            base_seed_values = np.asarray([values[name] for values in base_values])
            hybrid_seed_values = np.asarray([values[name] for values in hybrid_values])
            differences = hybrid_seed_values - base_seed_values
            low, high = np.quantile(bootstrapped[name], [0.025, 0.975])
            rows.append(
                {
                    "scope": scope,
                    "n_test": len(indexes),
                    "metric": name,
                    "difference_definition": "xgb(base+analytic) minus xgb(base)",
                    "base_mean_over_seeds": float(np.mean(base_seed_values)),
                    "base_seed_std": float(np.std(base_seed_values, ddof=1)),
                    "hybrid_mean_over_seeds": float(np.mean(hybrid_seed_values)),
                    "hybrid_seed_std": float(np.std(hybrid_seed_values, ddof=1)),
                    "difference_mean_over_seeds": float(np.mean(differences)),
                    "difference_seed_std": float(np.std(differences, ddof=1)),
                    "paired_bootstrap_ci_low": float(low),
                    "paired_bootstrap_ci_high": float(high),
                    "bootstrap_resamples": 2000,
                    "training_seeds": 5,
                }
            )
    write_csv(pd.DataFrame(rows), RESULTS_DIR / "paired_comparison.csv", provenance)


def _t_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, float("nan"), float("nan")
    margin = float(
        t.ppf(0.975, values.size - 1)
        * np.std(values, ddof=1)
        / np.sqrt(values.size)
    )
    return mean, mean - margin, mean + margin


def _write_feature_comparison(test, layouts, constants, y_true, provenance):
    rows = []
    for seed in range(5):
        base = _predict_model_seed(test, layouts, constants, "base", "all", seed)
        hybrid = _predict_model_seed(
            test, layouts, constants, "base_analytic", "all", seed
        )
        rng = np.random.default_rng(1000 + seed)
        base_rho, hybrid_rho, delta_rho = [], [], []
        base_top1, hybrid_top1, delta_top1 = [], [], []
        for _ in range(1000):
            indexes = rng.integers(0, len(y_true), size=len(y_true))
            truth = y_true[indexes]
            base_sample, hybrid_sample = base[indexes], hybrid[indexes]
            base_rho.append(spearman(truth, base_sample))
            hybrid_rho.append(spearman(truth, hybrid_sample))
            delta_rho.append(hybrid_rho[-1] - base_rho[-1])
            base_ratio = top_k_latency_ratio(truth, np.argsort(base_sample), 1)
            hybrid_ratio = top_k_latency_ratio(truth, np.argsort(hybrid_sample), 1)
            base_top1.append(base_ratio)
            hybrid_top1.append(hybrid_ratio)
            delta_top1.append(hybrid_ratio - base_ratio)

        base_point = spearman(y_true, base)
        hybrid_point = spearman(y_true, hybrid)
        base_ratio = top_k_latency_ratio(y_true, np.argsort(base), 1)
        hybrid_ratio = top_k_latency_ratio(y_true, np.argsort(hybrid), 1)
        rows.append(
            {
                "seed": seed,
                "base_spearman": base_point,
                "base_spearman_ci_low": float(np.quantile(base_rho, 0.025)),
                "base_spearman_ci_high": float(np.quantile(base_rho, 0.975)),
                "hybrid_spearman": hybrid_point,
                "hybrid_spearman_ci_low": float(np.quantile(hybrid_rho, 0.025)),
                "hybrid_spearman_ci_high": float(np.quantile(hybrid_rho, 0.975)),
                "delta_spearman_hybrid_minus_base": hybrid_point - base_point,
                "delta_spearman_ci_low": float(np.quantile(delta_rho, 0.025)),
                "delta_spearman_ci_high": float(np.quantile(delta_rho, 0.975)),
                "base_top_1_latency_ratio": base_ratio,
                "base_top_1_ci_low": float(np.quantile(base_top1, 0.025)),
                "base_top_1_ci_high": float(np.quantile(base_top1, 0.975)),
                "hybrid_top_1_latency_ratio": hybrid_ratio,
                "hybrid_top_1_ci_low": float(np.quantile(hybrid_top1, 0.025)),
                "hybrid_top_1_ci_high": float(np.quantile(hybrid_top1, 0.975)),
                "delta_top_1_hybrid_minus_base": hybrid_ratio - base_ratio,
                "delta_top_1_ci_low": float(np.quantile(delta_top1, 0.025)),
                "delta_top_1_ci_high": float(np.quantile(delta_top1, 0.975)),
            }
        )
    write_csv(
        pd.DataFrame(rows),
        RESULTS_DIR / "feature_comparison_seeds.csv",
        provenance,
    )
    summary = []
    for metric in (
        "base_spearman",
        "hybrid_spearman",
        "delta_spearman_hybrid_minus_base",
        "base_top_1_latency_ratio",
        "hybrid_top_1_latency_ratio",
        "delta_top_1_hybrid_minus_base",
    ):
        mean, low, high = _t_interval([row[metric] for row in rows])
        if metric in ("base_top_1_latency_ratio", "hybrid_top_1_latency_ratio"):
            low = max(1.0, low)
        summary.append(
            {
                "metric": metric,
                "mean_over_seeds": mean,
                "ci95_low": low,
                "ci95_high": high,
                "seed_count": len(rows),
            }
        )
    write_csv(
        pd.DataFrame(summary),
        RESULTS_DIR / "feature_comparison_summary.csv",
        provenance,
    )


def _write_shuffled_label_null(train, test, layouts, constants, y_true, scopes, provenance):
    with (RESULTS_DIR / "training_grid.json").open(encoding="utf-8") as source:
        grid = json.load(source)["best"]

    raw_rows = []
    first_global_hybrid = None
    for feature_set, include_derived in (
        ("base", False),
        ("base_analytic", True),
    ):
        x_train = _model_features(train, layouts, include_derived, constants)
        x_test = _model_features(test, layouts, include_derived, constants)
        config = grid[feature_set]
        for shuffle_mode in ("global", "within_sketch"):
            for seed in range(20):
                rng = np.random.default_rng(seed)
                labels = train["median_ms"].to_numpy(dtype=float)
                if shuffle_mode == "global":
                    shuffled = rng.permutation(labels)
                else:
                    shuffled = labels.copy()
                    sketch_ids = train["sketch_id"].to_numpy(dtype=int)
                    for sketch_id in np.unique(sketch_ids):
                        indexes = np.flatnonzero(sketch_ids == sketch_id)
                        shuffled[indexes] = rng.permutation(labels[indexes])

                model = train_xgb(
                    x_train,
                    shuffled,
                    max_depth=config["max_depth"],
                    learning_rate=config["learning_rate"],
                    n_estimators=config["n_estimators"],
                    seed=seed,
                )
                prediction = np.exp(model.predict(x_test))
                if (
                    shuffle_mode == "global"
                    and feature_set == "base_analytic"
                    and seed == 0
                ):
                    first_global_hybrid = prediction
                for scope, indexes in scopes:
                    values = _metric_values(y_true[indexes], prediction[indexes])
                    raw_rows.append(
                        {
                            "shuffle_mode": shuffle_mode,
                            "model": "xgb(base+analytic)"
                            if include_derived
                            else "xgb(base)",
                            "scope": scope,
                            "seed": seed,
                            "spearman": values["spearman"],
                            "fastest_decile_spearman": values[
                                "fastest_decile_spearman"
                            ],
                            "max_depth": config["max_depth"],
                            "learning_rate": config["learning_rate"],
                            "n_estimators": config["n_estimators"],
                        }
                    )

    raw = pd.DataFrame(raw_rows)
    summary_rows = []
    group_columns = ("shuffle_mode", "model", "scope")
    scope_sizes = {name: len(indexes) for name, indexes in scopes}
    for group_values, group in raw.groupby(list(group_columns), sort=False):
        shuffle_mode, model_name, scope = group_values
        summary = {
            "shuffle_mode": shuffle_mode,
            "model": model_name,
            "scope": scope,
            "n_shuffles": len(group),
            "seeds": "0-19",
            "n_train": len(train),
            "n_test": scope_sizes[scope],
            "max_depth": int(group["max_depth"].iloc[0]),
            "learning_rate": float(group["learning_rate"].iloc[0]),
            "n_estimators": int(group["n_estimators"].iloc[0]),
        }
        for metric in ("spearman", "fastest_decile_spearman"):
            values = group[metric].to_numpy(dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1))
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float(np.max(values))
        summary_rows.append(summary)
    write_csv(
        pd.DataFrame(summary_rows),
        RESULTS_DIR / "shuffled_label_null.csv",
        provenance,
    )

    if first_global_hybrid is not None:
        pooled = _metric_values(y_true, first_global_hybrid)
        write_csv(
            pd.DataFrame(
                [
                    {
                        "training_label_order": "global permutation; seed 0",
                        "spearman": pooled["spearman"],
                        "fastest_decile_spearman": pooled[
                            "fastest_decile_spearman"
                        ],
                        "n_train": len(train),
                        "n_test": len(test),
                    }
                ]
            ),
            RESULTS_DIR / "shuffled_label_check.csv",
            provenance,
        )

    print(
        "Shuffle design: global permutes TRAIN labels across all sketches; "
        "within_sketch permutes labels separately per sketch. sketch_id is "
        "included in both feature variants."
    )
    print(
        "Leakage checks: split overlap validation; features and analytic score "
        "use schedule/layout rather than latency labels; sketch_id is explicit."
    )
    global_pooled = raw[
        (raw["shuffle_mode"] == "global") & (raw["scope"] == "pooled")
    ]
    possible_leak = False
    for model_name, group in global_pooled.groupby("model", sort=False):
        for metric in ("spearman", "fastest_decile_spearman"):
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if abs(mean) > 2 * standard_error:
                possible_leak = True
            print(
                f"Global-shuffle {model_name} {metric}: mean={mean:.4f}, "
                f"std={np.std(values, ddof=1):.4f}, 2SE={2 * standard_error:.4f}"
            )
    if possible_leak:
        print(
            "WARNING: global shuffled-label result exceeds 2 standard errors "
            "from zero; possible leak requires investigation."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    provenance = capture_provenance(ROOT, allow_dirty=args.allow_dirty)

    frame = load_pool()
    splits = load_splits()
    validate_splits(splits, set(frame["id"].astype(int)))
    layouts = load_layouts()
    train_ids = set(int(value) for value in splits["train"])
    test_ids = set(int(value) for value in splits["test"])
    train = frame[frame["id"].astype(int).isin(train_ids)].reset_index(drop=True)
    test = frame[frame["id"].astype(int).isin(test_ids)].reset_index(drop=True)
    if test.empty:
        raise ValueError("The test split contains no modelable measurements")
    hand_set, fitted = _read_constants()
    y_true = test["median_ms"].to_numpy(dtype=float)

    base_seed_predictions = [
        _predict_model_seed(test, layouts, fitted, "base", "all", seed)
        for seed in range(5)
    ]
    hybrid_seed_predictions = [
        _predict_model_seed(test, layouts, fitted, "base_analytic", "all", seed)
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scopes = list(_scope_indices(test))
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
    write_csv(
        pd.DataFrame(score_rows),
        RESULTS_DIR / "evaluation.csv",
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
    _paired_comparison(
        y_true,
        base_seed_predictions,
        hybrid_seed_predictions,
        scopes,
        provenance,
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
        true_top_1 = set(true_order[:1].tolist())
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
                recall_top_1 = len(true_top_1 & drafted)
                draft_only_ratio = _draft_only_ratio(
                    scoped_truth, draft_score, requested_k
                )
                ratio = _draft_verify_ratio(
                    scoped_truth, draft_score, verify_score, requested_k
                )
                bootstrap = []
                rng = np.random.default_rng(
                    50_000 + scope_index * 100 + requested_k
                )
                for _ in range(1000):
                    sample = rng.integers(0, len(indexes), size=len(indexes))
                    bootstrap.append(
                        _draft_verify_ratio(
                            scoped_truth[sample],
                            draft_score[sample],
                            verify_score[sample],
                            requested_k,
                        )
                    )
                low, high = np.quantile(bootstrap, [0.025, 0.975])
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
                        "draft_recall_at_K_top_1_count": recall_top_1,
                        "draft_recall_at_K_top_1_fraction": float(recall_top_1),
                        "draft_only_top_1_latency_ratio": draft_only_ratio,
                        "verify_only_top_1_latency_ratio": _draft_only_ratio(
                            scoped_truth, verify_score, len(indexes)
                        ),
                        "best_1": ratio,
                        "top_1_latency_ratio": ratio,
                        "top_1_chosen_over_best_latency_ratio": ratio,
                        "best_1_ci_low": float(low),
                        "best_1_ci_high": float(high),
                        "top_1_ci_low": float(low),
                        "top_1_ci_high": float(high),
                    }
                )
    write_csv(
        pd.DataFrame(draft_rows),
        RESULTS_DIR / "draft_verify.csv",
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
    write_csv(
        pd.DataFrame(ablation_rows),
        RESULTS_DIR / "ablations.csv",
        provenance,
    )

    noise_rows = []
    for scope, indexes in scopes:
        pairs = []
        for row in test.iloc[indexes].itertuples(index=False):
            times = np.asarray(row.repeats_ms, dtype=float)
            midpoint = len(times) // 2
            if midpoint and len(times) - midpoint:
                pairs.append(
                    (
                        float(row.median_ms),
                        float(np.median(times[:midpoint])),
                        float(np.median(times[midpoint:])),
                    )
                )
        pairs.sort(key=lambda item: item[0])
        first = np.asarray([item[1] for item in pairs])
        second = np.asarray([item[2] for item in pairs])
        decile_n = max(2, int(np.ceil(len(pairs) * 0.1))) if len(pairs) >= 2 else len(pairs)
        noise_rows.append(
            {
                "scope": scope,
                "repeat_half_spearman": float(spearmanr(first, second).statistic)
                if len(pairs) > 1
                else float("nan"),
                "n_repeat_pairs": len(pairs),
                "top_decile_n": decile_n,
                "top_decile_repeat_half_spearman": float(
                    spearmanr(first[:decile_n], second[:decile_n]).statistic
                )
                if decile_n > 1
                else float("nan"),
            }
        )
    write_csv(pd.DataFrame(noise_rows), RESULTS_DIR / "noise_floor.csv", provenance)

    learning_rows = []
    for scope, indexes in scopes:
        scoped_test = test.iloc[indexes].reset_index(drop=True)
        scoped_truth = y_true[indexes]
        for size_name, subset_ids in splits["nested_train"].items():
            for seed in range(5):
                prediction = _predict_model_seed(
                    scoped_test, layouts, fitted, "base_analytic", size_name, seed
                )
                learning_rows.append(
                    {
                        "scope": scope,
                        "train_size": len(subset_ids),
                        "subset": size_name,
                        "seed": seed,
                        "spearman": spearman(scoped_truth, prediction),
                        "top_1_latency_ratio": top_k_latency_ratio(
                            scoped_truth, np.argsort(prediction), 1
                        ),
                    }
                )
    learning = pd.DataFrame(learning_rows)
    write_csv(learning, RESULTS_DIR / "learning_curve.csv", provenance)
    learning_summary = []
    for (scope, train_size, subset), group in learning.groupby(
        ["scope", "train_size", "subset"], sort=False
    ):
        rho_mean, rho_low, rho_high = _t_interval(group["spearman"])
        ratio_mean, ratio_low, ratio_high = _t_interval(group["top_1_latency_ratio"])
        ratio_low = max(1.0, ratio_low)
        learning_summary.append(
            {
                "scope": scope,
                "train_size": train_size,
                "subset": subset,
                "seed_count": len(group),
                "spearman_mean": rho_mean,
                "spearman_ci95_low": rho_low,
                "spearman_ci95_high": rho_high,
                "top_1_latency_ratio_mean": ratio_mean,
                "top_1_ci95_low": ratio_low,
                "top_1_ci95_high": ratio_high,
            }
        )
    write_csv(
        pd.DataFrame(learning_summary),
        RESULTS_DIR / "learning_curve_summary.csv",
        provenance,
    )

    _write_feature_comparison(test, layouts, fitted, y_true, provenance)
    _write_shuffled_label_null(
        train, test, layouts, fitted, y_true, scopes, provenance
    )

    all_data = load_pool(include_failures=True)
    write_csv(
        all_data["status"]
        .value_counts()
        .rename_axis("status")
        .reset_index(name="count"),
        RESULTS_DIR / "failure_rates.csv",
        provenance,
    )
    print(
        f"Evaluated {len(test)} held-out schedules "
        f"({max(2, int(np.ceil(len(test) * 0.1)))} in the fastest decile); "
        f"repeat-half Spearman={noise_rows[0]['repeat_half_spearman']:.3f}, "
        f"fastest-decile repeat-half Spearman="
        f"{noise_rows[0]['top_decile_repeat_half_spearman']:.3f}"
    )


if __name__ == "__main__":
    main()
