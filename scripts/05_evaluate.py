
#!/usr/bin/env python3
"""Evaluate all rankers on the held-out test split only."""
from __future__ import annotations

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
    best_k,
    fastest_decile_spearman,
    spearman,
    top_k_latency_ratio,
    top_k_recall,
)
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


def _predict_models(frame, layouts, constants, feature_set, size_name="all"):
    predictions = [
        _predict_model_seed(
            frame, layouts, constants, feature_set, size_name, seed
        )
        for seed in range(5)
    ]
    return np.mean(predictions, axis=0)


def _bootstrap_metrics(y_true, prediction, seed=0, samples=1000):
    rng = np.random.default_rng(seed)
    values = {
        "spearman": [],
        "fastest_decile_spearman": [],
        "top_10_recall_in_30": [],
        **{
            f"{metric}_{k}": []
            for metric in ("top_k_latency_ratio", "best_k")
            for k in (1, 5, 10)
        },
    }
    for _ in range(samples):
        indexes = rng.integers(0, len(y_true), size=len(y_true))
        truth, score = y_true[indexes], prediction[indexes]
        ranked = np.argsort(score, kind="stable")
        observed = {
            "spearman": spearman(truth, score),
            "fastest_decile_spearman": fastest_decile_spearman(truth, score),
            "top_10_recall_in_30": top_k_recall(truth, score, 10, 30),
        }
        for k in (1, 5, 10):
            observed[f"top_k_latency_ratio_{k}"] = top_k_latency_ratio(
                truth, ranked, min(k, len(truth))
            )
            observed[f"best_k_{k}"] = best_k(truth, ranked, min(k, len(truth)))
        for name, value in observed.items():
            if np.isfinite(value):
                values[name].append(float(value))
    return {
        name: np.quantile(samples, [0.025, 0.975]).tolist()
        if samples
        else [float("nan"), float("nan")]
        for name, samples in values.items()
    }


def _score_row(scope, name, y_true, prediction, seed=0):
    rank = np.argsort(prediction, kind="stable")
    intervals = _bootstrap_metrics(y_true, prediction, seed=seed)
    metrics = {
        "scope": scope,
        "scorer": name,
        "n_test": len(y_true),
        "fastest_decile_n": max(2, int(np.ceil(len(y_true) * 0.1))),
        "top_10_recall_true_k": min(10, len(y_true)),
        "top_10_recall_predicted_k": min(30, len(y_true)),
        "top_k_ratio_definition": "mean selected latency / fastest test latency; lower is better",
        "spearman": spearman(y_true, prediction),
        "fastest_decile_spearman": fastest_decile_spearman(y_true, prediction),
        "fastest_decile_spearman_ci_low": intervals["fastest_decile_spearman"][0],
        "fastest_decile_spearman_ci_high": intervals["fastest_decile_spearman"][1],
        "top_10_recall_in_30": top_k_recall(y_true, prediction, 10, 30),
        "top_10_recall_ci_low": intervals["top_10_recall_in_30"][0],
        "top_10_recall_ci_high": intervals["top_10_recall_in_30"][1],
    }
    metrics["spearman_ci_low"] = intervals["spearman"][0]
    metrics["spearman_ci_high"] = intervals["spearman"][1]
    for k in (1, 5, 10):
        metrics[f"top_{k}_latency_ratio"] = top_k_latency_ratio(y_true, rank, k)
        metrics[f"best_{k}"] = best_k(y_true, rank, k)
        metrics[f"top_{k}_ci_low"] = intervals[f"top_k_latency_ratio_{k}"][0]
        metrics[f"top_{k}_ci_high"] = intervals[f"top_k_latency_ratio_{k}"][1]
        metrics[f"best_{k}_ci_low"] = intervals[f"best_k_{k}"][0]
        metrics[f"best_{k}_ci_high"] = intervals[f"best_k_{k}"][1]
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


def _write_feature_comparison(test, layouts, constants, y_true):
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
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "feature_comparison_seeds.csv", index=False
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
    pd.DataFrame(summary).to_csv(
        RESULTS_DIR / "feature_comparison_summary.csv", index=False
    )


def _write_shuffled_label_check(train, test, layouts, constants, y_true):
    with (RESULTS_DIR / "training_grid.json").open(encoding="utf-8") as source:
        best = json.load(source)["best"]["base_analytic"]
    x_train = _model_features(train, layouts, True, constants)
    x_test = _model_features(test, layouts, True, constants)
    shuffled = train["median_ms"].to_numpy(dtype=float).copy()
    np.random.default_rng(0).shuffle(shuffled)
    model = train_xgb(
        x_train,
        shuffled,
        max_depth=best["max_depth"],
        learning_rate=best["learning_rate"],
        n_estimators=best["n_estimators"],
        seed=0,
    )
    prediction = np.exp(model.predict(x_test))
    pd.DataFrame(
        [
            {
                "training_label_order": "permuted once; fixed seed 0",
                "spearman": spearman(y_true, prediction),
                "fastest_decile_spearman": fastest_decile_spearman(
                    y_true, prediction
                ),
                "n_train": len(train),
                "n_test": len(test),
            }
        ]
    ).to_csv(RESULTS_DIR / "shuffled_label_check.csv", index=False)


def main() -> None:
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

    rng = np.random.default_rng(0)
    predictions = {
        "random": rng.random(len(test)),
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
        "xgb(base)": _predict_models(test, layouts, fitted, "base"),
        "xgb(base+analytic)": _predict_models(test, layouts, fitted, "base_analytic"),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scopes = list(_scope_indices(test))
    score_rows = []
    for scope_index, (scope, indexes) in enumerate(scopes):
        for scorer_index, (name, prediction) in enumerate(predictions.items()):
            score_rows.append(
                _score_row(
                    scope,
                    name,
                    y_true[indexes],
                    prediction[indexes],
                    seed=scope_index * 100 + scorer_index,
                )
            )
    pd.DataFrame(score_rows).to_csv(RESULTS_DIR / "evaluation.csv", index=False)

    draft_predictions = predictions["analytic(fitted)"]
    verify_predictions = predictions["xgb(base+analytic)"]
    draft_rows = []
    for scope_index, (scope, indexes) in enumerate(scopes):
        scoped_truth = y_true[indexes]
        draft_score = draft_predictions[indexes]
        verify_score = verify_predictions[indexes]
        for requested_k in (32, 64, 128, 256):
            bootstrap = []
            rng = np.random.default_rng(scope_index * 100 + requested_k)
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
            ratio = _draft_verify_ratio(
                scoped_truth, draft_score, verify_score, requested_k
            )
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            draft_rows.append(
                {
                    "scope": scope,
                    "n_test": len(indexes),
                    "K": requested_k,
                    "candidates": min(requested_k, len(indexes)),
                    "best_1": ratio,
                    "top_1_latency_ratio": ratio,
                    "top_1_chosen_over_best_latency_ratio": ratio,
                    "best_1_ci_low": float(low),
                    "best_1_ci_high": float(high),
                    "top_1_ci_low": float(low),
                    "top_1_ci_high": float(high),
                }
            )
    pd.DataFrame(draft_rows).to_csv(RESULTS_DIR / "draft_verify.csv", index=False)

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
    pd.DataFrame(ablation_rows).to_csv(RESULTS_DIR / "ablations.csv", index=False)

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
    pd.DataFrame(noise_rows).to_csv(RESULTS_DIR / "noise_floor.csv", index=False)

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
    learning.to_csv(RESULTS_DIR / "learning_curve.csv", index=False)
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
    pd.DataFrame(learning_summary).to_csv(
        RESULTS_DIR / "learning_curve_summary.csv", index=False
    )

    _write_feature_comparison(test, layouts, fitted, y_true)
    _write_shuffled_label_check(train, test, layouts, fitted, y_true)

    all_data = load_pool(include_failures=True)
    all_data["status"].value_counts().rename_axis("status").reset_index(name="count").to_csv(
        RESULTS_DIR / "failure_rates.csv", index=False
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
