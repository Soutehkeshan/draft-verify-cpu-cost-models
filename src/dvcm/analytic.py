"""A small Pruner-style additive compute and memory cost model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from dvcm.features import derived


@dataclass(frozen=True)
class AnalyticConstants:
    peak_flops: float = 4.0e11
    bandwidth_bytes_s: float = 1.0e11
    l1_bytes: float = 128 * 1024
    l2_bytes: float = 16 * 1024 * 1024
    lane_width: int = 4
    num_cores: int = 4
    vector_registers: int = 32

    def __post_init__(self) -> None:
        for name in (
            "peak_flops",
            "bandwidth_bytes_s",
            "l1_bytes",
            "l2_bytes",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("lane_width", "num_cores", "vector_registers"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Penalties:
    vector: bool = True
    parallel: bool = True
    cache: bool = True
    register: bool = True


def _capacity_penalty(capacity_bytes: float, footprint_bytes: float) -> float:
    return min(1.0, capacity_bytes / max(footprint_bytes, 1.0))


def _penalty_terms_from_descriptors(
    descriptors: dict[str, float],
    constants: AnalyticConstants,
    penalties: Penalties,
) -> dict[str, float]:
    factors = {
        "vector": (
            descriptors["vector_lane_utilisation"] if penalties.vector else 1.0
        ),
        "parallel": (
            descriptors["parallel_utilisation"] if penalties.parallel else 1.0
        ),
        "register": (
            descriptors["register_utilisation"] if penalties.register else 1.0
        ),
        "cache_l1": (
            _capacity_penalty(constants.l1_bytes, descriptors["tile_level_2_bytes"])
            if penalties.cache
            else 1.0
        ),
        "cache_l2": (
            _capacity_penalty(constants.l2_bytes, descriptors["tile_level_1_bytes"])
            if penalties.cache
            else 1.0
        ),
    }
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in factors.values()):
        raise ValueError("penalty terms must be finite values in [0, 1]")
    return factors


def penalty_terms(
    position: np.ndarray,
    layout: dict[str, Any],
    constants: AnalyticConstants = AnalyticConstants(),
    penalties: Penalties = Penalties(),
) -> dict[str, float]:
    """Return bounded compute and cache factors, with 1.0 meaning no penalty."""
    descriptors = derived(
        position,
        {
            **layout,
            "vector_lanes": constants.lane_width,
            "num_cores": constants.num_cores,
            "vector_registers": constants.vector_registers,
        },
    )
    return _penalty_terms_from_descriptors(descriptors, constants, penalties)


def latency_seconds(
    position: np.ndarray,
    layout: dict[str, Any],
    constants: AnalyticConstants = AnalyticConstants(),
    penalties: Penalties = Penalties(),
) -> float:
    """Estimate Pruner-style additive compute and memory time in seconds."""
    descriptors = derived(
        position,
        {
            **layout,
            "vector_lanes": constants.lane_width,
            "num_cores": constants.num_cores,
            "vector_registers": constants.vector_registers,
        },
    )
    factors = _penalty_terms_from_descriptors(descriptors, constants, penalties)
    compute_penalty = (
        factors["vector"] * factors["parallel"] * factors["register"]
    )
    memory_penalty = factors["cache_l1"] * factors["cache_l2"]
    compute_time = descriptors["flop_count"] / (
        constants.peak_flops * compute_penalty
    )
    memory_time = descriptors["bytes_moved"] / (
        constants.bandwidth_bytes_s * memory_penalty
    )
    return float(compute_time + memory_time)


def fit_constants(
    positions: list[np.ndarray],
    layouts: list[dict[str, Any]],
    latencies_ms: np.ndarray,
    initial: AnalyticConstants = AnalyticConstants(),
    penalties: Penalties = Penalties(),
) -> AnalyticConstants:
    """Fit peak FLOP/s and DRAM bandwidth against TRAIN log-latency values."""
    from scipy.optimize import least_squares

    observations = np.asarray(latencies_ms, dtype=float)
    if (
        observations.size == 0
        or len(positions) != observations.size
        or len(layouts) != observations.size
        or not np.all(np.isfinite(observations))
        or np.any(observations <= 0)
    ):
        raise ValueError("latencies_ms must contain positive finite training measurements")

    names = ("peak_flops", "bandwidth_bytes_s")
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

    lower = np.log([1e7, 1e6])
    upper = np.log([1e15, 1e13])
    fitted = least_squares(
        residual,
        initial_values,
        bounds=(lower, upper),
        max_nfev=1000,
    )
    values = initial.to_dict()
    values.update(dict(zip(names, np.exp(fitted.x))))
    return AnalyticConstants(**values)
