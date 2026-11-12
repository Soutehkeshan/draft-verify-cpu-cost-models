
#!/usr/bin/env python3
"""Evaluate all rankers on the held-out test split only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.analytic import AnalyticConstants, Penalties, latency_seconds
from dvcm.features import features
from dvcm.metrics import best_k, spearman, top_k_latency_ratio
from dvcm.study import RESULTS_DIR, load_layouts, load_pool, load_splits


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
    result = {"spearman_ci95": []}
    spearman_values = []
    for _ in range(samples):
        indexes = rng.integers(0, len(y_true), size=len(y_true))
        value = spearman(y_true[indexes], prediction[indexes])
        if np.isfinite(value):
            spearman_values.append(value)
    if spearman_values:
        result["spearman_ci95"] = np.quantile(
            spearman_values, [0.025, 0.975]
        ).tolist()
    for k in (1, 5, 10):
        top_values = []
        best_values = []
        for _ in range(samples):
            indexes = rng.integers(0, len(y_true), size=len(y_true))
            truth = y_true[indexes]
            ranked = np.argsort(prediction[indexes])
            top_values.append(top_k_latency_ratio(truth, ranked, min(k, len(truth))))
            best_values.append(best_k(truth, ranked, min(k, len(truth))))
        result[k] = {
            "top_k_latency_ratio_ci95": np.quantile(top_values, [0.025, 0.975]).tolist(),
            "best_k_ci95": np.quantile(best_values, [0.025, 0.975]).tolist(),
        }
    return result


def _score_row(name, y_true, prediction):
    rank = np.argsort(prediction)
    metrics = {
        "scorer": name,
        "spearman": spearman(y_true, prediction),
    }
    intervals = _bootstrap_metrics(y_true, prediction)
    metrics["spearman_ci_low"] = intervals["spearman_ci95"][0]
    metrics["spearman_ci_high"] = intervals["spearman_ci95"][1]
    for k in (1, 5, 10):
        metrics[f"top_{k}_latency_ratio"] = top_k_latency_ratio(y_true, rank, k)
        metrics[f"best_{k}"] = best_k(y_true, rank, k)
        metrics[f"top_{k}_ci_low"] = intervals[k]["top_k_latency_ratio_ci95"][0]
        metrics[f"top_{k}_ci_high"] = intervals[k]["top_k_latency_ratio_ci95"][1]
        metrics[f"best_{k}_ci_low"] = intervals[k]["best_k_ci95"][0]
        metrics[f"best_{k}_ci_high"] = intervals[k]["best_k_ci95"][1]
    return metrics


def main() -> None:
    frame = load_pool()
    splits = load_splits()
    layouts = load_layouts()
    test_ids = set(int(value) for value in splits["test"])
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
    pd.DataFrame(
        [_score_row(name, y_true, prediction) for name, prediction in predictions.items()]
    ).to_csv(RESULTS_DIR / "evaluation.csv", index=False)

    draft_predictions = predictions["analytic(fitted)"]
    verify_predictions = predictions["xgb(base+analytic)"]
    draft_rows = []
    for requested_k in (32, 64, 128, 256):
        draft_ids = np.argsort(draft_predictions)[: min(requested_k, len(test))]
        verified_id = int(draft_ids[np.argmin(verify_predictions[draft_ids])])
        rng = np.random.default_rng(0)
        boot_values = []
        for _ in range(1000):
            indexes = rng.integers(0, len(y_true), size=len(y_true))
            drafted = np.argsort(draft_predictions[indexes])[
                : min(requested_k, len(indexes))
            ]
            verified = int(drafted[np.argmin(verify_predictions[indexes][drafted])])
            boot_values.append(
                float(
                    y_true[indexes][verified]
                    / np.min(y_true[indexes])
                )
            )
        confidence_interval = np.quantile(boot_values, [0.025, 0.975])
        draft_rows.append(
            {
                "K": requested_k,
                "candidates": len(draft_ids),
                "best_1": float(y_true[verified_id] / np.min(y_true)),
                "top_1_latency_ratio": float(y_true[verified_id] / np.min(y_true)),
                "best_1_ci_low": float(confidence_interval[0]),
                "best_1_ci_high": float(confidence_interval[1]),
                "top_1_ci_low": float(confidence_interval[0]),
                "top_1_ci_high": float(confidence_interval[1]),
            }
        )
    pd.DataFrame(draft_rows).to_csv(RESULTS_DIR / "draft_verify.csv", index=False)

    ablation_rows = []
    for model_name, prediction in (
        ("xgb_base_features", predictions["xgb(base)"]),
        ("xgb_base_plus_derived_and_analytic", predictions["xgb(base+analytic)"]),
    ):
        ablation_rows.append(
            {
                "ablation": model_name,
                "spearman": spearman(y_true, prediction),
                "top_1_latency_ratio": top_k_latency_ratio(
                    y_true, np.argsort(prediction), 1
                ),
                "best_1": best_k(y_true, np.argsort(prediction), 1),
            }
        )
    for removed in ("vector", "parallel", "cache"):
        penalties = Penalties(**{name: name != removed for name in ("vector", "parallel", "cache")})
        score = np.asarray(
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
        ablation_rows.append(
            {
                "ablation": f"without_{removed}",
                "spearman": spearman(y_true, score),
                "top_1_latency_ratio": top_k_latency_ratio(y_true, np.argsort(score), 1),
                "best_1": best_k(y_true, np.argsort(score), 1),
            }
        )
    pd.DataFrame(ablation_rows).to_csv(RESULTS_DIR / "ablations.csv", index=False)

    noise_truth = []
    noise_prediction = []
    for row in test.itertuples(index=False):
        times = np.asarray(row.repeats_ms, dtype=float)
        midpoint = len(times) // 2
        if midpoint and len(times) - midpoint:
            noise_truth.append(float(np.median(times[:midpoint])))
            noise_prediction.append(float(np.median(times[midpoint:])))
    noise = float(spearmanr(noise_truth, noise_prediction).statistic) if len(noise_truth) > 1 else float("nan")
    pd.DataFrame([{"repeat_half_spearman": noise, "n": len(noise_truth)}]).to_csv(
        RESULTS_DIR / "noise_floor.csv", index=False
    )

    learning_rows = []
    for size_name, subset_ids in splits["nested_train"].items():
        train_size = len(subset_ids)
        for seed in range(5):
            prediction = _predict_model_seed(
                test, layouts, fitted, "base_analytic", size_name, seed
            )
            learning_rows.append(
                {
                    "train_size": train_size,
                    "subset": size_name,
                    "seed": seed,
                    "spearman": spearman(y_true, prediction),
                    "top_1_latency_ratio": top_k_latency_ratio(
                        y_true, np.argsort(prediction), 1
                    ),
                }
            )
    pd.DataFrame(learning_rows).to_csv(RESULTS_DIR / "learning_curve.csv", index=False)

    all_data = load_pool(include_failures=True)
    all_data["status"].value_counts().rename_axis("status").reset_index(name="count").to_csv(
        RESULTS_DIR / "failure_rates.csv", index=False
    )
    print(f"Evaluated {len(test)} held-out schedules; repeat-half Spearman={noise:.3f}")


if __name__ == "__main__":
    main()
