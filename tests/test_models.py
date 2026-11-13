import numpy as np

from dvcm.analytic import AnalyticConstants, fit_constants, latency_seconds
from dvcm.config import LEGACY_TARGET, load_target_config
from dvcm.learned import predict_latency, train_xgb


def _layout():
    return {
        "dim": 11,
        "tiles": [
            {"n": 4, "slice": [0, 4], "total_exp": 9, "max_inner_exp": 6},
            {"n": 4, "slice": [4, 8], "total_exp": 9, "max_inner_exp": 6},
            {"n": 2, "slice": [8, 10], "total_exp": 9, "max_inner_exp": 6},
        ],
        "categoricals": [{"index": 10, "candidates": [0, 16, 64, 512]}],
        "parallel_tile_levels": {"i": 2, "j": 2},
        **load_target_config(LEGACY_TARGET).feature_settings(),
    }


def _positions():
    return [
        np.array(value, dtype=float)
        for value in (
            (9, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0),
            (0, 3, 3, 3, 0, 3, 3, 3, 3, 6, 0),
            (0, 0, 3, 6, 0, 0, 3, 6, 3, 6, 0),
            (2, 2, 2, 3, 2, 2, 2, 3, 6, 3, 0),
            (0, 4, 0, 5, 4, 0, 0, 5, 5, 4, 0),
        )
    ]


def test_analytic_constants_fit_positive_training_measurements():
    layout = _layout()
    layouts = [layout] * len(_positions())
    positions = _positions()
    initial = AnalyticConstants(**load_target_config(LEGACY_TARGET).analytic_constants())
    truth = AnalyticConstants(
        **{
            **initial.to_dict(),
            "peak_flops": 2.5e11,
            "bandwidth_bytes_s": 8.0e10,
        }
    )
    latencies = np.array(
        [latency_seconds(position, layout, truth) * 1000 for position in positions]
    )

    fitted = fit_constants(positions, layouts, latencies, initial=initial)

    assert np.isfinite([fitted.peak_flops, fitted.bandwidth_bytes_s]).all()
    assert fitted.peak_flops > 0
    assert fitted.bandwidth_bytes_s > 0
    assert fitted.l1_bytes == initial.l1_bytes
    assert fitted.lane_width == initial.lane_width
    fitted_error = np.mean(
        np.square(
            np.log(
                [
                    latency_seconds(position, layout, fitted) * 1000
                    for position in positions
                ]
            )
            - np.log(latencies)
        )
    )
    initial_error = np.mean(
        np.square(
            np.log(
                [
                    latency_seconds(position, layout, initial) * 1000
                    for position in positions
                ]
            )
            - np.log(latencies)
        )
    )
    assert fitted_error <= initial_error


def test_xgboost_training_returns_positive_latency_predictions():
    x_train = np.arange(40, dtype=float).reshape(20, 2)
    y_train = np.exp(x_train[:, 0] / 20.0)
    model = train_xgb(
        x_train,
        y_train,
        max_depth=2,
        learning_rate=0.1,
        n_estimators=5,
        seed=0,
    )

    predictions = predict_latency(model, x_train[:4])

    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
    assert np.all(predictions > 0)
