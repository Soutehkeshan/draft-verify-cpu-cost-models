
#!/usr/bin/env python3
"""Generate publication-ready PDF summaries without seaborn."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.analytic import AnalyticConstants, latency_seconds
from dvcm.features import features
from dvcm.metrics import spearman
from dvcm.study import RESULTS_DIR, load_layouts, load_pool, load_splits


def main() -> None:
    plots = RESULTS_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    evaluation = pd.read_csv(RESULTS_DIR / "evaluation.csv")
    learning = pd.read_csv(RESULTS_DIR / "learning_curve.csv")
    draft = pd.read_csv(RESULTS_DIR / "draft_verify.csv")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].axis("off")
    table_data = evaluation[["scorer", "spearman", "top_1_latency_ratio", "best_5"]]
    axes[0, 0].table(
        cellText=np.round(table_data.iloc[:, 1:].to_numpy(dtype=float), 3),
        rowLabels=table_data["scorer"].tolist(),
        colLabels=table_data.columns[1:].tolist(),
        loc="center",
    )
    axes[0, 0].set_title("Test-set results")

    correlation_axis = axes[0, 1].twinx()
    colors = plt.get_cmap("tab10")
    for seed, group in learning.groupby("seed", sort=True):
        group = group.sort_values("train_size")
        color = colors(seed)
        axes[0, 1].plot(
            group["train_size"],
            group["top_1_latency_ratio"],
            marker="o",
            color=color,
            label=f"Top-1 seed {seed}",
        )
        correlation_axis.plot(
            group["train_size"],
            group["spearman"],
            linestyle="--",
            color=color,
            alpha=0.8,
            label=f"Spearman seed {seed}",
        )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Training schedules")
    axes[0, 1].set_ylabel("Top-1 latency ratio")
    correlation_axis.set_ylabel("Spearman correlation")
    axes[0, 1].set_title("Learning curve")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(draft["K"], draft["best_1"], marker="o", label="Best-1")
    axes[1, 0].plot(draft["K"], draft["top_1_latency_ratio"], marker="x", label="Top-1")
    axes[1, 0].set_xlabel("Draft size K")
    axes[1, 0].set_ylabel("Latency ratio (lower is better)")
    axes[1, 0].set_title("Draft then verify")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.25)

    canary_path = RESULTS_DIR / "canary.csv"
    if canary_path.exists():
        canary = pd.read_csv(canary_path)
        canary = canary[canary["status"].isin(("ok", "slow_skip"))]
        axes[1, 1].scatter(canary["batch_id"], canary["median_ms"], alpha=0.5)
        axes[1, 1].set_title("Canary drift")
        axes[1, 1].set_xlabel("Batch")
        axes[1, 1].set_ylabel("Latency (ms)")
    else:
        axes[1, 1].text(0.5, 0.5, "Canary data unavailable", ha="center")
        axes[1, 1].set_axis_off()

    figure.tight_layout()
    figure.savefig(plots / "overview.pdf")
    plt.close(figure)

    with (RESULTS_DIR / "constants.json").open(encoding="utf-8") as source:
        constants = AnalyticConstants(**json.load(source)["constants"])
    layouts = load_layouts()
    pool = load_pool()
    test_ids = set(load_splits()["test"])
    test = pool[pool["id"].astype(int).isin(test_ids)].reset_index(drop=True)
    matrices = []
    for row in test.itertuples(index=False):
        layout = layouts[int(row.sketch_id)]
        position = np.asarray(row.vector, dtype=float)
        values = features(position, int(row.sketch_id), layout, True)
        matrices.append(
            np.append(
                values,
                latency_seconds(position, layout, constants) * 1000,
            )
        )
    prediction = []
    for seed in range(5):
        model = XGBRegressor()
        model.load_model(RESULTS_DIR / "models" / f"base_analytic_nall_seed{seed}.json")
        prediction.append(np.exp(model.predict(np.asarray(matrices))))
    predicted = np.mean(prediction, axis=0)
    measured = test["median_ms"].to_numpy(dtype=float)
    scatter, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(measured, predicted, alpha=0.65)
    low = min(measured.min(), predicted.min())
    high = max(measured.max(), predicted.max())
    axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Measured latency (ms)")
    axis.set_ylabel("Predicted latency (ms)")
    axis.set_title(f"Measured vs predicted (Spearman {spearman(measured, predicted):.3f})")
    scatter.tight_layout()
    scatter.savefig(plots / "measured_vs_predicted.pdf")
    plt.close(scatter)
    print(f"Plots written to {plots}")


if __name__ == "__main__":
    main()
