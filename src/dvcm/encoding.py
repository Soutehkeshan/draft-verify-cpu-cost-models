
"""Encode MetaSchedule trace decisions as integer vectors, and back.

Problem size (512x512 matmul, per sketch):
    i tiling : 4 exponents of 2, sum = 9, innermost <= 5  -> 3 free dims, 200 valid
    j tiling : 4 exponents of 2, sum = 9, innermost <= 5  -> 3 free dims, 200 valid
    k tiling : 2 exponents of 2, sum = 9, innermost <= 5  -> 1 free dim,    6 valid
    unroll   : 1 categorical index, 0..3                   -> 1 dim,         4 valid
The vector stores all 11 numbers (4 + 4 + 2 + 1); repair enforces the sum constraint,
so only 8 are effectively free. The sketch is not in the vector; it is recorded
separately as sketch_id.
"""
import math

import numpy as np

TOTAL_EXP = 9          # 512 = 2**9; derive from the loop extent, don't hard-code later
MAX_INNER_EXP = 5      # max_innermost_factor = 32 = 2**5; read from the trace instead
NUM_UNROLL = 4         # candidates in the SampleCategorical instruction

# Default slices for the current workload. Actual layouts are derived from traces.
I_SLICE = slice(0, 4)
J_SLICE = slice(4, 8)
K_SLICE = slice(8, 10)
UNROLL_IDX = 10
DIM = 11
_DEFAULT_LAYOUT_CACHE = None


def _default_layout():
    """The current 512x512 matmul layout, used when no trace is supplied."""
    tiles = []
    for tile_slice in (I_SLICE, J_SLICE, K_SLICE):
        n = tile_slice.stop - tile_slice.start
        tiles.append({
            "inst": None,
            "n": n,
            "extent": 2**TOTAL_EXP,
            "total_exp": TOTAL_EXP,
            "max_inner_factor": 2**MAX_INNER_EXP,
            "max_inner_exp": MAX_INNER_EXP,
            "slice": tile_slice,
            "valid_exponents": _valid_tile_exponents(n, TOTAL_EXP, MAX_INNER_EXP),
        })
    return {
        "tiles": tiles,
        "categoricals": [{"inst": None, "candidates": tuple(range(NUM_UNROLL)), "index": UNROLL_IDX}],
        "dim": DIM,
    }


def _get_default_layout():
    """Build and cache the current workload's default layout once."""
    global _DEFAULT_LAYOUT_CACHE
    if _DEFAULT_LAYOUT_CACHE is None:
        _DEFAULT_LAYOUT_CACHE = _default_layout()
    return _DEFAULT_LAYOUT_CACHE


def _sampling_instructions(trace):
    """Return supported sampling instructions in trace order."""
    tile_insts = []
    categorical_insts = []
    ordered_insts = []
    for inst in trace.insts:
        kind = inst.kind.name
        if kind == "SamplePerfectTile":
            tile_insts.append(inst)
            ordered_insts.append(inst)
        elif kind == "SampleCategorical":
            categorical_insts.append(inst)
            ordered_insts.append(inst)
        elif kind.startswith("Sample"):
            raise ValueError(f"Unsupported sampling instruction in encoded vector: {kind}")
    return tile_insts, categorical_insts, ordered_insts


def _parallel_tile_levels(trace, tiles):
    split_insts = [inst for inst in trace.insts if inst.kind.name == "Split"]
    if len(split_insts) != len(tiles):
        raise ValueError("Could not match sampled tiles to their split loops")
    if len(tiles) != 3 or tiles[0]["n"] < 2 or tiles[1]["n"] < 2:
        raise ValueError("Parallel-loop inference requires the i, j, k SSRSRS tiling")

    j_split = split_insts[1]
    placements = [
        inst
        for inst in trace.insts
        if inst.kind.name == "ReverseComputeAt"
    ]
    if len(placements) > 1:
        raise ValueError("Expected at most one cache-write placement")
    if not placements:
        if any(inst.kind.name == "CacheWrite" for inst in trace.insts):
            raise ValueError("CacheWrite trace is missing its ReverseComputeAt placement")
        return {"i": 2, "j": 2}

    target_loop = placements[0].inputs[1]
    for level, output in enumerate(j_split.outputs):
        if target_loop == output:
            return {"i": level + 1, "j": level + 1}
    raise ValueError("Cache-write placement does not target a sampled j loop")


def describe_space(design_space_trace):
    """Describe encoded sampling instructions in a design-space trace.

    Tile extent is inferred from the product of its recorded perfect-tile decision;
    the trace stores the sampling instruction's loop RV but not that loop's extent.
    The returned layout records instruction, tile constraints, position slices, and
    categorical candidates in trace order.
    """
    trace = getattr(design_space_trace, "trace", design_space_trace)
    tiles = []
    categoricals = []
    offset = 0
    _, _, ordered_insts = _sampling_instructions(trace)
    for inst in ordered_insts:
        if inst.kind.name == "SamplePerfectTile":
            n = int(inst.attrs[0])
            max_inner_factor = int(inst.attrs[1])
            if n < 1 or max_inner_factor < 1:
                raise ValueError("SamplePerfectTile requires positive n and max_innermost_factor")
            decision = trace.get_decision(inst)
            if decision is None:
                raise ValueError("SamplePerfectTile instruction has no recorded decision")
            factors = tuple(int(factor) for factor in decision)
            if len(factors) != n or any(factor <= 0 for factor in factors):
                raise ValueError(f"Invalid SamplePerfectTile decision: {factors}")
            extent = math.prod(factors)
            total_exp = int(round(math.log2(extent)))
            if 2**total_exp != extent:
                raise ValueError(f"Tile extent {extent} is not a power of two")
            max_inner_exp = int(math.floor(math.log2(max_inner_factor)))
            tiles.append({
                "inst": inst,
                "n": n,
                "extent": extent,
                "total_exp": total_exp,
                "max_inner_factor": max_inner_factor,
                "max_inner_exp": max_inner_exp,
                "slice": slice(offset, offset + n),
                "valid_exponents": _valid_tile_exponents(n, total_exp, max_inner_exp),
            })
            offset += n
        else:
            candidates = tuple(int(candidate) for candidate in inst.attrs[0])
            if not candidates:
                raise ValueError("SampleCategorical instruction has no candidates")
            decision = trace.get_decision(inst)
            if decision is None or not 0 <= int(decision) < len(candidates):
                raise ValueError(
                    f"Invalid SampleCategorical decision {decision} for candidates {candidates}"
                )
            categoricals.append({"inst": inst, "candidates": candidates, "index": offset})
            offset += 1

    if not tiles:
        raise ValueError("Trace contains no SamplePerfectTile instructions")
    if not categoricals:
        raise ValueError("Trace contains no SampleCategorical instructions")
    return {
        "tiles": tiles,
        "categoricals": categoricals,
        "dim": offset,
        "parallel_tile_levels": _parallel_tile_levels(trace, tiles),
    }


def repair_tile(exps, total=TOTAL_EXP, max_inner=MAX_INNER_EXP):
    """Map one loop's continuous exponents to the NEAREST valid integer tiling.

    Valid = non-negative integers, sum == total, last <= max_inner.

    Projects onto the bounded integer simplex by minimizing squared distance from
    the clipped exponents. Ties are resolved by choosing the first coordinate.
    """
    values = np.asarray(exps, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("exps must be a non-empty one-dimensional sequence")
    if total < 0 or max_inner < 0:
        raise ValueError("total and max_inner must be non-negative")
    if not np.all(np.isfinite(values)):
        raise ValueError("exps must contain only finite values")

    target = np.clip(values, 0, total)
    target[-1] = min(target[-1], max_inner)
    result = np.rint(target).astype(int)
    limits = np.full(result.size, total, dtype=int)
    limits[-1] = min(total, max_inner)
    result = np.minimum(result, limits)

    while result.sum() < total:
        available = np.flatnonzero(result < limits)
        if available.size == 0:
            raise ValueError("No valid tiling satisfies the requested total and bounds")
        # Adding a unit where target-result is largest minimizes squared error.
        idx = available[np.argmax(target[available] - result[available])]
        result[idx] += 1
    while result.sum() > total:
        available = np.flatnonzero(result > 0)
        # Removing a unit where result-target is largest minimizes squared error.
        idx = available[np.argmax(result[available] - target[available])]
        result[idx] -= 1
    return result


def repair(position, layout=None):
    """Return a valid copy of a whole encoded position (tilings plus categories)."""
    layout = _get_default_layout() if layout is None else layout
    position = np.asarray(position, dtype=float)
    if position.shape != (layout["dim"],):
        raise ValueError(f"Expected position shape {(layout['dim'],)}, got {position.shape}")
    if not np.all(np.isfinite(position)):
        raise ValueError("position must contain only finite values")

    repaired = position.copy()
    for tile in layout["tiles"]:
        tile_slice = tile["slice"]
        repaired[tile_slice] = repair_tile(
            repaired[tile_slice], tile["total_exp"], tile["max_inner_exp"]
        )
    for categorical in layout["categoricals"]:
        idx = categorical["index"]
        repaired[idx] = np.clip(
            np.rint(repaired[idx]), 0, len(categorical["candidates"]) - 1
        )
    return repaired


def position_to_decisions(position, design_space_trace, layout=None):
    """Return a trace with sample decisions replaced by the repaired position."""
    layout = describe_space(design_space_trace) if layout is None else layout
    trace = getattr(design_space_trace, "trace", design_space_trace)
    tile_insts, categorical_insts, _ = _sampling_instructions(trace)
    if len(tile_insts) != len(layout["tiles"]) or len(categorical_insts) != len(
        layout["categoricals"]
    ):
        raise ValueError("Trace sampling instructions do not match the cached layout")
    position = repair(position, layout)
    updated_trace = trace
    for tile, inst in zip(layout["tiles"], tile_insts):
        exponents = position[tile["slice"]].astype(int)
        factors = [2**int(exponent) for exponent in exponents]
        updated_trace = updated_trace.with_decision(inst, factors, remove_postproc=True)
    for categorical, inst in zip(layout["categoricals"], categorical_insts):
        candidate_idx = int(position[categorical["index"]])
        updated_trace = updated_trace.with_decision(
            inst, candidate_idx, remove_postproc=True
        )
    return updated_trace


def decisions_to_position(trace, layout=None):
    """Convert recorded tile factors and categorical decisions to an encoded position."""
    trace = getattr(trace, "trace", trace)
    layout = describe_space(trace) if layout is None else layout
    tile_insts, categorical_insts, _ = _sampling_instructions(trace)
    if len(tile_insts) != len(layout["tiles"]) or len(categorical_insts) != len(
        layout["categoricals"]
    ):
        raise ValueError("Trace sampling instructions do not match the cached layout")
    position = np.zeros(layout["dim"], dtype=float)
    for tile, inst in zip(layout["tiles"], tile_insts):
        factors = trace.get_decision(inst)
        if factors is None or len(factors) != tile["n"]:
            raise ValueError("Trace has no complete SamplePerfectTile decision")
        exponents = []
        for factor in factors:
            factor = int(factor)
            exponent = int(round(np.log2(factor))) if factor > 0 else -1
            if factor <= 0 or 2**exponent != factor:
                raise ValueError(f"Tile factor {factor} is not a positive power of two")
            exponents.append(exponent)
        if sum(exponents) != tile["total_exp"]:
            raise ValueError("Tile factors do not match the recorded loop extent")
        if exponents[-1] > tile["max_inner_exp"]:
            raise ValueError("Innermost tile factor exceeds its configured maximum")
        position[tile["slice"]] = exponents

    for categorical, inst in zip(layout["categoricals"], categorical_insts):
        decision = trace.get_decision(inst)
        if decision is None:
            raise ValueError("Trace has no SampleCategorical decision")
        candidate_idx = int(decision)
        if not 0 <= candidate_idx < len(categorical["candidates"]):
            raise ValueError(
                f"Categorical decision index {candidate_idx} is out of range for "
                f"{categorical['candidates']}"
            )
        position[categorical["index"]] = candidate_idx
    return position


def layout_to_json(layout, sketch_id):
    """Serialize the structural fields of a describe_space() layout."""
    return {
        "sketch_id": int(sketch_id),
        "tiles": [
            {
                "n": int(tile["n"]),
                "extent": int(tile["extent"]),
                "total_exp": int(tile["total_exp"]),
                "max_inner_exp": int(tile["max_inner_exp"]),
                "max_inner_factor": int(tile["max_inner_factor"]),
                "slice": [int(tile["slice"].start), int(tile["slice"].stop)],
            }
            for tile in layout["tiles"]
        ],
        "categoricals": [
            {
                "index": int(category["index"]),
                "candidates": [int(candidate) for candidate in category["candidates"]],
            }
            for category in layout["categoricals"]
        ],
        "dim": int(layout["dim"]),
        "parallel_tile_levels": {
            "i": int(layout["parallel_tile_levels"]["i"]),
            "j": int(layout["parallel_tile_levels"]["j"]),
        },
    }


STRUCTURAL_LAYOUT_KEYS = ("sketch_id", "tiles", "categoricals", "dim", "parallel_tile_levels")


def structural_layout(layout_json):
    """Return only the schedule-space fields of a serialized layout (no target values)."""
    return {key: layout_json[key] for key in STRUCTURAL_LAYOUT_KEYS}


def tile_choice_count(tile):
    """Number of valid exponent compositions for one tile group (live or JSON layout)."""
    choices = tile.get("valid_exponents")
    if choices is None:
        choices = _valid_tile_exponents(
            int(tile["n"]), int(tile["total_exp"]), int(tile["max_inner_exp"])
        )
    return len(choices)


def space_size(layout):
    """Number of valid encoded positions implied by a live or JSON layout."""
    return math.prod(
        [tile_choice_count(tile) for tile in layout["tiles"]]
        + [len(category["candidates"]) for category in layout["categoricals"]]
    )


def _valid_tile_exponents(n, total, max_inner):
    """Enumerate bounded exponent compositions for uniform valid-tile sampling."""
    compositions = []

    def visit(prefix, remaining, parts_left):
        if parts_left == 1:
            if 0 <= remaining <= max_inner:
                compositions.append(prefix + [remaining])
            return
        for value in range(min(total, remaining) + 1):
            visit(prefix + [value], remaining - value, parts_left - 1)

    visit([], total, n)
    return compositions


def random_position(rng, layout=None):
    """Sample uniformly from valid encoded positions for the described layout."""
    layout = _get_default_layout() if layout is None else layout
    position = np.zeros(layout["dim"], dtype=float)
    for tile in layout["tiles"]:
        choices = tile.get("valid_exponents")
        if choices is None:
            choices = _valid_tile_exponents(tile["n"], tile["total_exp"], tile["max_inner_exp"])
            tile["valid_exponents"] = choices
        if not choices:
            raise ValueError("No valid tile exponents for the described layout")
        position[tile["slice"]] = choices[int(rng.integers(len(choices)))]
    for categorical in layout["categoricals"]:
        position[categorical["index"]] = rng.integers(len(categorical["candidates"]))
    return position
