
"""XGBoost log-latency models and deterministic data preparation."""
from __future__ import annotations

from typing import Any

import numpy as np


def train_xgb(
    x_train: np.ndarray,
    y_train_ms: np.ndarray,
    *,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    n_estimators: int = 300,
    seed: int = 0,
) -> Any:
    """Fit an XGBoost regressor to the logarithm of measured latency."""
    from xgboost import XGBRegressor

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
    )
    model.fit(x_train, np.log(np.maximum(y_train_ms, 1e-12)))
    return model


def predict_latency(model: Any, features: np.ndarray) -> np.ndarray:
    """Convert model log predictions to milliseconds."""
    return np.exp(np.asarray(model.predict(features), dtype=float))
