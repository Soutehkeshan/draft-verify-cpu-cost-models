
"""A small Pruner-style roofline cost model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dvcm.features import derived


@dataclass(frozen=True)
class AnalyticConstants:
    peak_flops: float = 1.0e12
    bandwidth_bytes_s: float = 6.8e10
    l1_bytes: float = 128 * 1024
    l2_bytes: float = 4 * 1024 * 1024
    l3_bytes: float = 16 * 1024 * 1024
    lane_width: int = 4
    num_cores: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Penalties:
    vector: bool = True
    parallel: bool = True
    cache: bool = True


def latency_seconds(
    position: np.ndarray,
    layout: dict[str, Any],
    constants: AnalyticConstants = AnalyticConstants(),
    penalties: Penalties = Penalties(),
) -> float:
    """Estimate compute plus memory time; each utilisation term is ablatable."""
    features = derived(
        position,
        {
            **layout,
            "vector_lanes": constants.lane_width,
            "num_cores": constants.num_cores,
        },
    )
    work_flops = features["flop_count"]
    bytes_moved = features["bytes_moved"]
    compute_util = 1.0
    memory_util = 1.0
    if penalties.vector:
        compute_util *= max(features["vector_lane_utilisation"], 1e-6)
    if penalties.parallel:
        compute_util *= max(features["parallel_utilisation"], 1e-6)
    if penalties.cache:
        footprint = features["tile_footprint_bytes"]
        cache_util = (
            0.3 * min(1.0, constants.l1_bytes / max(footprint, 1.0))
            + 0.3 * min(1.0, constants.l2_bytes / max(footprint, 1.0))
            + 0.4 * min(1.0, constants.l3_bytes / max(footprint, 1.0))
        )
        memory_util *= max(cache_util, 1e-6)
    compute_time = work_flops / (constants.peak_flops * compute_util)
    memory_time = bytes_moved / (constants.bandwidth_bytes_s * memory_util)
    return float(compute_time + memory_time)


def fit_constants(
    positions: list[np.ndarray],
    layouts: list[dict[str, Any]],
    latencies_ms: np.ndarray,
    initial: AnalyticConstants = AnalyticConstants(),
    penalties: Penalties = Penalties(),
) -> AnalyticConstants:
    """Fit positive hardware constants against TRAIN log-latency values."""
    from scipy.optimize import least_squares

    observations = np.asarray(latencies_ms, dtype=float)
    if (
        observations.size == 0
        or len(positions) != observations.size
        or len(layouts) != observations.size
        or np.any(observations <= 0)
    ):
        raise ValueError("latencies_ms must contain positive training measurements")

    names = ("peak_flops", "bandwidth_bytes_s", "l1_bytes", "l2_bytes")
    initial_values = np.log([getattr(initial, name) for name in names])

    def residual(log_values: np.ndarray) -> np.ndarray:
        constants = AnalyticConstants(
            **{
                **initial.to_dict(),
                **dict(zip(names, np.exp(log_values))),
            }
        )
        prediction = np.array(
            [
                latency_seconds(position, layout, constants, penalties)
                for position, layout in zip(positions, layouts)
            ]
        ) * 1000.0
        return np.log(np.maximum(prediction, 1e-12)) - np.log(observations)

    lower = np.log([1e7, 1e6, 1024, 1024])
    upper = np.log([1e15, 1e13, 1e9, 1e10])
    fitted = least_squares(
        residual,
        initial_values,
        bounds=(lower, upper),
        max_nfev=1000,
    )
    values = initial.to_dict()
    values.update(dict(zip(names, np.exp(fitted.x))))
    return AnalyticConstants(**values)
