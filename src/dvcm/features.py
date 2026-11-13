"""Feature construction for encoded MetaSchedule positions."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

POSITION_NAMES = (
    "i0", "i1", "i2", "i3",
    "j0", "j1", "j2", "j3",
    "k0", "k1", "unroll",
)
DERIVED_NAMES = (
    "tile_level_2_bytes",
    "tile_level_3_bytes",
    "bytes_moved",
    "vector_lane_utilisation",
    "parallel_utilisation",
)
ALL_DERIVED_NAMES = (
    "tile_level_0_bytes",
    "tile_level_1_bytes",
    "tile_level_2_bytes",
    "tile_level_3_bytes",
    "bytes_moved",
    "flop_count",
    "vector_lane_utilisation",
    "parallel_utilisation",
    "register_utilisation",
)
_LEVEL_COUNTS = (4, 4, 2)
_BYTES_PER_ELEMENT = 4
# Layout key -> TargetConfig field supplying it.
TARGET_FEATURE_FIELDS = {
    "vector_lanes": "simd_width_f32",
    "num_cores": "num_cores",
    "vector_registers": "vector_registers",
}


def bind_target_settings(layout: dict[str, Any], target: Any) -> dict[str, Any]:
    """Return a copy of layout whose hardware values come only from the target profile.

    Values baked into older layout files are dropped when the profile leaves them
    unset, so derived() then refuses instead of silently reusing another target.
    """
    bound = dict(layout)
    for name, field in TARGET_FEATURE_FIELDS.items():
        value = getattr(target, field)
        if value is None:
            bound.pop(name, None)
        else:
            bound[name] = int(value)
    return bound


def _tile_slice(tile: dict[str, Any]) -> slice:
    value = tile["slice"]
    if isinstance(value, slice):
        return value
    return slice(int(value[0]), int(value[1]))


def _categorical_index(categorical: dict[str, Any]) -> int:
    return int(categorical["index"])


def feature_names(
    layout: dict[str, Any] | None = None, include_derived: bool = True
) -> tuple[str, ...]:
    """Return names matching features(), with sketch_id outside the encoded position."""
    del layout
    names = POSITION_NAMES + ("sketch_id",)
    return names + DERIVED_NAMES if include_derived else names


def derived(position: np.ndarray, layout: dict[str, Any]) -> dict[str, float]:
    """Compute tile working sets, DRAM-boundary traffic, work, and utilization."""
    position = np.asarray(position, dtype=float)
    tiles = layout["tiles"]
    if len(tiles) != len(_LEVEL_COUNTS):
        raise ValueError(f"Expected i, j, and k tile groups; got {len(tiles)}")
    if position.shape != (int(layout["dim"]),):
        raise ValueError(f"Expected position shape {(int(layout['dim']),)}, got {position.shape}")
    if not np.all(np.isfinite(position)):
        raise ValueError("position must contain only finite values")

    factors_by_group: list[tuple[int, ...]] = []
    for tile, level_count in zip(tiles, _LEVEL_COUNTS):
        if int(tile["n"]) != level_count:
            raise ValueError(
                f"Expected {level_count} tile levels for this loop group; got {tile['n']}"
            )
        raw_exponents = position[_tile_slice(tile)]
        exponents = np.rint(raw_exponents).astype(int)
        if (
            exponents.size != level_count
            or np.any(exponents < 0)
            or not np.array_equal(exponents, raw_exponents)
            or int(exponents.sum()) != int(tile["total_exp"])
            or exponents[-1] > int(tile["max_inner_exp"])
        ):
            raise ValueError("position contains an invalid tile exponent group")
        factors_by_group.append(tuple(2**int(exponent) for exponent in exponents))

    i_factors, j_factors, k_factors = factors_by_group
    m, n, k = (math.prod(factors) for factors in factors_by_group)
    # SSRSRS order is i0,j0,i1,j1,k0,i2,j2,k1,i3,j3.
    level_footprints = []
    for level in range(_LEVEL_COUNTS[0]):
        i_extent = int(np.prod(i_factors[level:], dtype=np.int64))
        j_extent = int(np.prod(j_factors[level:], dtype=np.int64))
        k_level = max(0, level - 1)
        k_extent = int(np.prod(k_factors[k_level:], dtype=np.int64))
        level_footprints.append(
            float(
                _BYTES_PER_ELEMENT
                * (i_extent * k_extent + k_extent * j_extent + i_extent * j_extent)
            )
        )

    categorical = layout["categoricals"][0]
    raw_candidate_idx = position[_categorical_index(categorical)]
    candidate_idx = int(round(raw_candidate_idx))
    candidates = categorical["candidates"]
    if candidate_idx != raw_candidate_idx:
        raise ValueError("position contains a non-integer unroll candidate index")
    if not 0 <= candidate_idx < len(candidates):
        raise ValueError("position contains an invalid unroll candidate index")
    missing_hardware = [
        name
        for name in ("vector_lanes", "num_cores", "vector_registers")
        if name not in layout or layout[name] is None
    ]
    if missing_hardware:
        raise ValueError(
            "Target feature settings are required: "
            + ", ".join(missing_hardware)
        )
    vector_lanes = int(layout["vector_lanes"])
    num_cores = int(layout["num_cores"])
    vector_registers = int(layout["vector_registers"])
    if vector_lanes < 1 or num_cores < 1 or vector_registers < 1:
        raise ValueError("vector_lanes, num_cores, and vector_registers must be positive")
    vector_util = min(j_factors[-1], vector_lanes) / vector_lanes
    parallel_levels = layout["parallel_tile_levels"]
    i_parallel_levels = int(parallel_levels["i"])
    j_parallel_levels = int(parallel_levels["j"])
    if (
        not 1 <= i_parallel_levels <= len(i_factors)
        or not 1 <= j_parallel_levels <= len(j_factors)
    ):
        raise ValueError("parallel_tile_levels must select valid i and j levels")
    parallel_tasks = math.prod(i_factors[:i_parallel_levels]) * math.prod(
        j_factors[:j_parallel_levels]
    )
    parallel_util = parallel_tasks / (
        math.ceil(parallel_tasks / num_cores) * num_cores
    )
    register_util = min(
        1.0,
        vector_registers * vector_lanes / (i_factors[-1] * j_factors[-1]),
    )
    level_1_regions = i_factors[0] * j_factors[0]
    bytes_moved = float(level_1_regions * level_footprints[1])
    values = {
        **{
            f"tile_level_{level}_bytes": value
            for level, value in enumerate(level_footprints)
        },
        "bytes_moved": bytes_moved,
        "flop_count": float(2 * m * n * k),
        "vector_lane_utilisation": float(vector_util),
        "parallel_utilisation": float(parallel_util),
        "register_utilisation": float(register_util),
    }
    if not np.all(np.isfinite(list(values.values()))):
        raise ValueError("derived features must be finite")
    return values


def features(
    position: np.ndarray,
    sketch_id: int,
    layout: dict[str, Any],
    include_derived: bool = True,
) -> np.ndarray:
    """Return encoded exponents, sketch identifier, and optional model features."""
    position = np.asarray(position, dtype=float)
    if position.shape != (len(POSITION_NAMES),):
        raise ValueError(f"Expected {len(POSITION_NAMES)} encoded values, got {position.shape}")
    values = np.append(position, float(sketch_id))
    if include_derived:
        descriptors = derived(position, layout)
        values = np.append(values, [descriptors[name] for name in DERIVED_NAMES])
    if not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    return values.astype(float)
