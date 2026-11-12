"""Feature construction for encoded MetaSchedule positions."""
from __future__ import annotations

from typing import Any

import numpy as np

POSITION_NAMES = (
    "i0", "i1", "i2", "i3",
    "j0", "j1", "j2", "j3",
    "k0", "k1", "unroll",
)
DERIVED_NAMES = (
    "i_tile_elements",
    "j_tile_elements",
    "k_tile_elements",
    "tile_level_0_bytes",
    "tile_level_1_bytes",
    "tile_level_2_bytes",
    "tile_level_3_bytes",
    "tile_footprint_bytes",
    "bytes_moved",
    "flop_count",
    "vector_lane_utilisation",
    "parallel_utilisation",
    "unroll_value",
)


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
    """Compute transparent tile and hardware-utilisation descriptors."""
    position = np.asarray(position, dtype=float)
    tiles = layout["tiles"]
    if len(tiles) != 3:
        raise ValueError(f"Expected i, j, and k tile groups; got {len(tiles)}")
    if position.shape != (int(layout["dim"]),):
        raise ValueError(f"Expected position shape {(int(layout['dim']),)}, got {position.shape}")
    if not np.all(np.isfinite(position)):
        raise ValueError("position must contain only finite values")

    factors_by_group = []
    for tile in tiles:
        tile_slice = _tile_slice(tile)
        raw_exponents = position[tile_slice]
        exponents = np.rint(raw_exponents).astype(int)
        if (
            np.any(exponents < 0)
            or not np.array_equal(exponents, raw_exponents)
            or int(exponents.sum()) != int(tile["total_exp"])
            or exponents[-1] > int(tile["max_inner_exp"])
        ):
            raise ValueError("position contains an invalid tile exponent group")
        factors_by_group.append(np.exp2(exponents).astype(int))

    i_factors, j_factors, k_factors = factors_by_group
    i_elements, j_elements, k_elements = (
        int(np.prod(values, dtype=np.int64))
        for values in factors_by_group
    )
    level_footprints = []
    for level in range(4):
        i_level = int(i_factors[level])
        j_level = int(j_factors[level])
        k_level = int(k_factors[level]) if level < len(k_factors) else 1
        level_footprints.append(
            float(4 * (i_level * k_level + k_level * j_level + i_level * j_level))
        )
    footprint_bytes = float(sum(level_footprints))
    candidates = layout["categoricals"][0]["candidates"]
    raw_candidate_idx = position[_categorical_index(layout["categoricals"][0])]
    candidate_idx = int(round(raw_candidate_idx))
    if candidate_idx != raw_candidate_idx:
        raise ValueError("position contains a non-integer unroll candidate index")
    if not 0 <= candidate_idx < len(candidates):
        raise ValueError("position contains an invalid unroll candidate index")
    unroll_value = float(candidates[candidate_idx])
    total_exp = sum(int(tile["total_exp"]) for tile in tiles)
    vector_lanes = float(layout.get("vector_lanes", 4))
    inner_factor = int(i_factors[-1])
    vector_util = min(inner_factor, vector_lanes) / vector_lanes
    outer_parallel = max(
        1,
        (2**int(tiles[0]["total_exp"]) // int(i_factors[0]))
        * (2**int(tiles[1]["total_exp"]) // int(j_factors[0])),
    )
    parallel_util = min(outer_parallel, int(layout.get("num_cores", 4))) / float(
        layout.get("num_cores", 4)
    )
    n = 2**int(tiles[0]["total_exp"])
    bytes_moved = 4.0 * (
        (n**3 / int(j_factors[-1]))
        + (n**3 / int(i_factors[-1]))
        + n**2
    )
    return {
        "i_tile_elements": float(i_elements),
        "j_tile_elements": float(j_elements),
        "k_tile_elements": float(k_elements),
        **{
            f"tile_level_{level}_bytes": value
            for level, value in enumerate(level_footprints)
        },
        "tile_footprint_bytes": footprint_bytes,
        "bytes_moved": bytes_moved,
        "flop_count": float(2 * 2**total_exp),
        "vector_lane_utilisation": float(vector_util),
        "parallel_utilisation": float(parallel_util),
        "unroll_value": unroll_value,
    }


def features(
    position: np.ndarray,
    sketch_id: int,
    layout: dict[str, Any],
    include_derived: bool = True,
) -> np.ndarray:
    """Return encoded exponents, sketch identifier, and optional derived features."""
    position = np.asarray(position, dtype=float)
    if position.shape != (len(POSITION_NAMES),):
        raise ValueError(f"Expected {len(POSITION_NAMES)} encoded values, got {position.shape}")
    values = np.append(position, float(sketch_id))
    if include_derived:
        values = np.append(values, list(derived(position, layout).values()))
    if not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    return values.astype(float)
