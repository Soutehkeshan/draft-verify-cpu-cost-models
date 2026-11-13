#!/usr/bin/env python3
"""Jetson Nano qualification steps. None of these steps collects a dataset.

Subcommands (outputs go to data/nano/; the M2 data in data/ is only read):
  parity      layout/design-space parity between the M2 collection and the Nano target
  loop-order  scheduled TIR per sketch under both targets; SSRSRS order check
  replay      stratified 150-schedule replay + pilot timing on the Nano (RPC)
  noise       20-schedule canary, 5 spaced rounds + a second RPC session (RPC)
  throughput  per-schedule time split and collection-time projections
  features    Nano-profile feature/penalty extraction over the replay positions
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tvm  # noqa: E402
from tvm.s_tir import meta_schedule as ms  # noqa: E402
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply  # noqa: E402

from dvcm.config import load_target_config, load_workload_config  # noqa: E402
from dvcm.encoding import (  # noqa: E402
    describe_space,
    layout_to_json,
    repair,
    space_size,
    structural_layout,
    tile_choice_count,
)
from dvcm.harness import HarnessConfig, _schedule_from_position, make_matmul  # noqa: E402
from dvcm.study import save_json  # noqa: E402

M2_DATA = ROOT / "data"
NANO_DATA = ROOT / "data" / "nano"
QUAL_DIR = NANO_DATA / "qualification"
RUN_META = NANO_DATA / "run_meta.json"
PILOT_REPORT = NANO_DATA / "pilot_report.json"
REPLAY_TABLE = QUAL_DIR / "replay_pilot.parquet"
RULES = "llvm"
POSTPROCS = "llvm"
SSRSRS = ("i_0", "j_0", "i_1", "j_1", "k_0", "i_2", "j_2", "k_1", "i_3", "j_3")


# --------------------------------------------------------------------------- helpers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    sources = [Path(__file__)] + sorted((ROOT / "src" / "dvcm").glob("*.py"))
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_paths": git("status", "--porcelain", "--untracked-files=all").splitlines(),
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tvm_version": str(tvm.__version__),
    }


def design_space(target: Any, rules: str, postprocs: str) -> tuple[list, list[dict]]:
    workload = load_workload_config(root=ROOT)
    context = ms.TuneContext(
        mod=make_matmul(workload.n),
        target=target,
        space_generator=PostOrderApply(sch_rules=rules, postprocs=postprocs),
        rand_state=workload.seed,
        num_threads=4,
    )
    sketches = context.generate_design_space()
    return sketches, [describe_space(sketch.trace) for sketch in sketches]


def nano_design_space() -> tuple[Any, list, list[dict]]:
    target = load_target_config("jetson_nano", ROOT).make_tvm_target()
    sketches, layouts = design_space(target, RULES, POSTPROCS)
    return target, sketches, layouts


def harness_config(target: Any, slow_threshold_ms: float = float("inf")) -> HarnessConfig:
    workload = load_workload_config(root=ROOT)
    return HarnessConfig(
        target=target,
        postprocessors=POSTPROCS,
        n=workload.n,
        repeats=workload.measurement["repeats"],
        number=workload.measurement["number"],
        min_repeat_ms=workload.measurement["min_repeat_ms"],
        seed=workload.seed,
        slow_threshold_ms=slow_threshold_ms,
    )


def load_run_meta() -> dict[str, Any]:
    with RUN_META.open(encoding="utf-8") as source:
        return json.load(source)


def update_run_meta(**fields: Any) -> None:
    meta = load_run_meta()
    meta.update(fields)
    save_json(RUN_META, meta)


def _m2_pool() -> pd.DataFrame:
    return pd.read_parquet(M2_DATA / "pool.parquet")


# --------------------------------------------------------------------------- step 1


def cmd_parity(_: argparse.Namespace) -> int:
    m2_profile = load_target_config("apple_m2", ROOT)
    nano_profile = load_target_config("jetson_nano", ROOT)
    m2_target = m2_profile.make_tvm_target()
    nano_target = nano_profile.make_tvm_target()
    print(f"M2 collection rules/postprocs (apple_m2 profile): "
          f"{m2_profile.schedule_rules!r}/{m2_profile.postprocessors!r}")
    print(f"Nano rules/postprocs used here: {RULES!r}/{POSTPROCS!r} "
          f"(jetson_nano profile fields: {nano_profile.schedule_rules!r}/"
          f"{nano_profile.postprocessors!r} -- collection gate)")
    m2_sketches, m2_layouts = design_space(
        m2_target, m2_profile.schedule_rules, m2_profile.postprocessors
    )
    nano_sketches, nano_layouts = design_space(nano_target, RULES, POSTPROCS)

    m2_rules = [str(rule) for rule in ms.ScheduleRule.create(m2_profile.schedule_rules)]
    nano_rules = [str(rule) for rule in ms.ScheduleRule.create(RULES)]
    m2_post = [str(p) for p in ms.Postproc.create(m2_profile.postprocessors)]
    nano_post = [str(p) for p in ms.Postproc.create(POSTPROCS)]

    sketch_reports = []
    stop_reasons = []
    m2_files = sorted(M2_DATA.glob("layout_*.json"))
    if not (len(m2_files) == len(m2_sketches) == len(nano_sketches)):
        stop_reasons.append(
            f"sketch count mismatch: files={len(m2_files)} m2={len(m2_sketches)} "
            f"nano={len(nano_sketches)}"
        )
    header = (f"{'sketch':>6} {'group':>5} {'n':>7} {'extent':>11} {'max_inner':>13} "
              f"{'choices':>11} {'unroll candidates':>38}")
    print(header)
    for sketch_id in range(min(len(m2_files), len(nano_sketches))):
        with (M2_DATA / f"layout_{sketch_id}.json").open(encoding="utf-8") as source:
            m2_file = json.load(source)
        m2_regen = layout_to_json(m2_layouts[sketch_id], sketch_id)
        nano_json = layout_to_json(nano_layouts[sketch_id], sketch_id)
        tiles = []
        for group, (m2_tile, nano_tile) in enumerate(zip(m2_file["tiles"], nano_json["tiles"])):
            row = {
                "group": "ijk"[group],
                "n": (m2_tile["n"], nano_tile["n"]),
                "extent": (m2_tile["extent"], nano_tile["extent"]),
                "max_inner_factor": (m2_tile["max_inner_factor"], nano_tile["max_inner_factor"]),
                "max_inner_exp": (m2_tile["max_inner_exp"], nano_tile["max_inner_exp"]),
                "valid_choices": (tile_choice_count(m2_tile), tile_choice_count(nano_tile)),
                "slice": (m2_tile["slice"], nano_tile["slice"]),
            }
            tiles.append(row)
            print(f"{sketch_id:>6} {row['group']:>5} {str(row['n']):>7} "
                  f"{str(row['extent']):>11} {str(row['max_inner_factor']):>13} "
                  f"{str(row['valid_choices']):>11}")
            for key in ("n", "max_inner_factor", "max_inner_exp", "extent", "slice"):
                if row[key][0] != row[key][1]:
                    stop_reasons.append(f"sketch {sketch_id} tile {row['group']} {key} {row[key]}")
        unroll = (
            [c["candidates"] for c in m2_file["categoricals"]],
            [c["candidates"] for c in nano_json["categoricals"]],
        )
        print(f"{sketch_id:>6} {'unrl':>5} {'':>7} {'':>11} {'':>13} {'':>11} "
              f"{str(unroll[0]) + ' | ' + str(unroll[1]):>38}")
        if len(m2_file["tiles"]) != len(nano_json["tiles"]):
            stop_reasons.append(f"sketch {sketch_id} tile-group count differs")
        if unroll[0] != unroll[1]:
            stop_reasons.append(f"sketch {sketch_id} unroll candidates differ {unroll}")
        report = {
            "sketch_id": sketch_id,
            "valid_positions_m2_layout_file": space_size(m2_file),
            "valid_positions_nano_design_space": space_size(nano_layouts[sketch_id]),
            "m2_file_matches_m2_regenerated": structural_layout(m2_file) == m2_regen,
            "m2_file_matches_nano": structural_layout(m2_file) == nano_json,
            "parallel_tile_levels": (m2_file["parallel_tile_levels"], nano_json["parallel_tile_levels"]),
            "dim": (m2_file["dim"], nano_json["dim"]),
            "trace_identical_m2_vs_nano_target": str(m2_sketches[sketch_id].trace)
            == str(nano_sketches[sketch_id].trace),
            "tiles": tiles,
            "unroll_candidates": unroll,
        }
        if not report["m2_file_matches_nano"]:
            stop_reasons.append(f"sketch {sketch_id} structural layout differs")
        sketch_reports.append(report)

    print()
    print(f"{'sketch':>6} {'valid (M2 layout file)':>24} {'valid (Nano design space)':>27} "
          f"{'file==M2 regen':>15} {'file==Nano':>11} {'trace M2==Nano':>15}")
    for item in sketch_reports:
        print(f"{item['sketch_id']:>6} {item['valid_positions_m2_layout_file']:>24,} "
              f"{item['valid_positions_nano_design_space']:>27,} "
              f"{str(item['m2_file_matches_m2_regenerated']):>15} "
              f"{str(item['m2_file_matches_nano']):>11} "
              f"{str(item['trace_identical_m2_vs_nano_target']):>15}")
    totals = (
        sum(item["valid_positions_m2_layout_file"] for item in sketch_reports),
        sum(item["valid_positions_nano_design_space"] for item in sketch_reports),
    )
    print(f"{'total':>6} {totals[0]:>24,} {totals[1]:>27,}")
    print()
    print(f"schedule rules identical (M2 profile vs Nano 'llvm'): {m2_rules == nano_rules}")
    for rule in nano_rules:
        print(f"  rule: {rule[:150]}")
    print(f"postprocessors identical: {m2_post == nano_post}")
    for post in nano_post:
        print(f"  postproc: {post}")
    if m2_rules != nano_rules:
        stop_reasons.append("schedule rule lists differ")
    if m2_post != nano_post:
        stop_reasons.append("postprocessor lists differ")

    written = []
    if not stop_reasons:
        for sketch_id, layout in enumerate(nano_layouts):
            path = NANO_DATA / f"layout_{sketch_id}.json"
            payload = layout_to_json(layout, sketch_id)
            if path.exists():
                with path.open(encoding="utf-8") as source:
                    if json.load(source) != payload:
                        stop_reasons.append(f"existing {path} differs from Nano design space")
                continue
            save_json(path, payload)
            written.append(str(path.relative_to(ROOT)))
    save_json(
        QUAL_DIR / "layout_parity.json",
        {
            "m2_target": str(m2_target),
            "nano_target": str(nano_target),
            "m2_rules_postprocs": [m2_profile.schedule_rules, m2_profile.postprocessors],
            "nano_rules_postprocs": [RULES, POSTPROCS],
            "rules": {"m2": m2_rules, "nano": nano_rules},
            "postprocessors": {"m2": m2_post, "nano": nano_post},
            "sketches": sketch_reports,
            "totals": {"m2_layout_files": totals[0], "nano_design_space": totals[1]},
            "stop_reasons": stop_reasons,
            "nano_layouts_written": written,
            "provenance": provenance(),
        },
    )
    if stop_reasons:
        print("STOP: " + "; ".join(stop_reasons))
        return 2
    print(f"PARITY OK; Nano layouts written: {written or 'already present (identical)'}")
    return 0


# --------------------------------------------------------------------------- step 2


def _loop_tokens(name: str) -> list[str]:
    return re.findall(r"([ijk]_\d)", name)


def _compute_loops(schedule: Any) -> list[dict[str, Any]]:
    block = schedule.get_sblock("C_update", func_name="main")
    loops = []
    for loop_rv in schedule.get_loops(block):
        loop = schedule.get(loop_rv)
        loops.append(
            {
                "name": loop.loop_var.name,
                "extent": int(loop.extent),
                "kind": str(loop.kind),
                "tokens": _loop_tokens(loop.loop_var.name),
            }
        )
    return loops


def _check_ssrsrs(loops: list[dict[str, Any]], position: np.ndarray, layout: dict) -> list[str]:
    errors = []
    tokens = [token for loop in loops for token in loop["tokens"]]
    if tuple(tokens) != SSRSRS:
        errors.append(f"loop token order {tokens} != {list(SSRSRS)}")
    factors = {}
    for tile, axis in zip(layout["tiles"], "ijk"):
        exponents = np.asarray(position[tile["slice"]], dtype=int)
        for level, exponent in enumerate(exponents):
            factors[f"{axis}_{level}"] = 2 ** int(exponent)
    for loop in loops:
        expected = math.prod(factors[token] for token in loop["tokens"])
        if loop["extent"] != expected:
            errors.append(f"{loop['name']} extent {loop['extent']} != {expected}")
    levels = layout["parallel_tile_levels"]
    parallel = [loop for loop in loops if "Parallel" in loop["kind"] or loop["kind"] == "1"]
    expected_parallel = SSRSRS[: levels["i"] + levels["j"]]
    if not parallel or tuple(parallel[0]["tokens"]) != expected_parallel:
        errors.append(f"parallel loop {parallel[:1]} != fused {list(expected_parallel)}")
    if not loops[-1]["tokens"] or loops[-1]["tokens"][-1] != "j_3":
        errors.append(f"innermost loop {loops[-1]['name']} is not j_3")
    return errors


def cmd_loop_order(args: argparse.Namespace) -> int:
    pool = _m2_pool().sort_values("id")
    targets = {
        "apple_m2": load_target_config("apple_m2", ROOT).make_tvm_target(),
        "jetson_nano": load_target_config("jetson_nano", ROOT).make_tvm_target(),
    }
    spaces = {name: design_space(target, RULES, POSTPROCS) for name, target in targets.items()}
    out_dir = QUAL_DIR / "loop_order"
    report = []
    failures = 0
    for sketch_id in sorted(pool["sketch_id"].unique()):
        candidates = pool[
            (pool["sketch_id"] == sketch_id)
            & pool["vector"].apply(lambda v: all(int(x) >= 1 for x in v[:10]))
        ]
        row = (candidates if not candidates.empty else pool[pool["sketch_id"] == sketch_id]).iloc[0]
        position = np.asarray(row["vector"], dtype=float)
        scripts = {}
        loops_by_target = {}
        for name, target in targets.items():
            sketches, layouts = spaces[name]
            schedule = _schedule_from_position(
                position, sketches[sketch_id], layouts[sketch_id], harness_config(target)
            )
            scripts[name] = schedule.mod.script()
            loops_by_target[name] = _compute_loops(schedule)
            path = out_dir / f"sketch{sketch_id}_{name}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(scripts[name], encoding="utf-8")
        layout = spaces["jetson_nano"][1][sketch_id]
        errors = _check_ssrsrs(loops_by_target["jetson_nano"], position, layout)
        identical = scripts["apple_m2"] == scripts["jetson_nano"]
        if not identical:
            errors.append("scheduled TIR differs between targets")
        failures += bool(errors)
        lib = tvm.compile(
            _schedule_from_position(
                position, spaces["jetson_nano"][0][sketch_id], layout,
                harness_config(targets["jetson_nano"]),
            ).mod,
            target=targets["jetson_nano"],
        )
        asm = lib.mod.inspect_source("asm") if hasattr(lib.mod, "inspect_source") else ""
        neon = {
            "fmla_v4s": len(re.findall(r"\bfmla\s+v\d+\.4s", asm)),
            "fmla_v2s": len(re.findall(r"\bfmla\s+v\d+\.2s", asm)),
            "fmadd_scalar": len(re.findall(r"\bfmadd\s+s\d+", asm)),
            "fmul_or_fadd_v4s": len(re.findall(r"\bf(?:mul|add)\s+v\d+\.4s", asm)),
            "fmul_or_fadd_v2s": len(re.findall(r"\bf(?:mul|add)\s+v\d+\.2s", asm)),
            "fmul_or_fadd_scalar": len(re.findall(r"\bf(?:mul|add)\s+s\d+", asm)),
            "asm_chars": len(asm),
        }
        print(f"=== sketch {sketch_id}: M2 pool id {int(row['id'])}, vector {list(map(int, row['vector']))}, "
              f"parallel_tile_levels {layout['parallel_tile_levels']}")
        print("C_update loop nest (outer -> inner), Nano target:")
        for loop in loops_by_target["jetson_nano"]:
            print(f"   {loop['name']:<28} extent={loop['extent']:<5} kind={loop['kind']}")
        print(f"   tokens: {[t for loop in loops_by_target['jetson_nano'] for t in loop['tokens']]}")
        print(f"   scheduled TIR identical M2 vs Nano target: {identical}")
        print(f"   Nano AArch64 asm: {neon}")
        print(f"   SSRSRS check: {'OK' if not errors else errors}")
        if args.print_tir:
            print(scripts["jetson_nano"])
        report.append(
            {
                "sketch_id": int(sketch_id),
                "m2_pool_id": int(row["id"]),
                "vector": [int(v) for v in row["vector"]],
                "loops": loops_by_target,
                "tir_identical_across_targets": identical,
                "nano_asm_counts": neon,
                "errors": errors,
                "tir_files": {
                    name: str((out_dir / f"sketch{sketch_id}_{name}.py").relative_to(ROOT))
                    for name in targets
                },
            }
        )
    save_json(QUAL_DIR / "loop_order.json", {"sketches": report, "provenance": provenance()})
    return 1 if failures else 0


# --------------------------------------------------------------------------- steps 3 + 4

REPLAY_SEED = 0
REPLAY_PER_CELL = 5  # 3 sketches x 10 M2-latency deciles x 5 = 150


class _ErrorCapture(logging.Handler):
    """Keep the harness's logged exception text for the current schedule."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            error = record.exc_info[1]
            text += f": {type(error).__name__}: {str(error).strip().splitlines()[-1][:400] if str(error).strip() else ''}"
        self.messages.append(text)


def select_replay_set(pool: pd.DataFrame) -> pd.DataFrame:
    """150 rows stratified by sketch and per-sketch M2 latency decile; fixed seed."""
    rng = np.random.default_rng(REPLAY_SEED)
    picks = []
    for sketch_id, group in pool.sort_values("id").groupby("sketch_id", sort=True):
        deciles = pd.qcut(group["median_ms"].rank(method="first"), 10, labels=False).to_numpy()
        for decile in range(10):
            cell = group[deciles == decile]
            for index in sorted(rng.choice(len(cell), size=REPLAY_PER_CELL, replace=False)):
                record = cell.iloc[int(index)]
                picks.append(
                    {
                        "id": int(record["id"]),
                        "sketch_id": int(sketch_id),
                        "vector": [int(v) for v in record["vector"]],
                        "m2_median_ms": float(record["median_ms"]),
                        "m2_status": str(record["status"]),
                        "m2_decile": int(decile),
                    }
                )
    replay = pd.DataFrame(picks)
    replay["run_order"] = rng.permutation(len(replay))
    return replay.sort_values("run_order").reset_index(drop=True)


def _checkpoint(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def _spearman(x: Any, y: Any) -> float | None:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return None
    return float(spearmanr(x, y).statistic)


def pilot_report(frame: pd.DataFrame, before: dict, after: dict) -> dict[str, Any]:
    from dvcm.device_state import frequency_residency, summarize_device_state

    ok = frame[frame["status"] == "ok"]
    medians = ok["median_ms"].to_numpy(dtype=float)
    pilot_median = float(np.median(medians))
    threshold = 20.0 * pilot_median
    per_sketch = {}
    for sketch_id, group in frame.groupby("sketch_id"):
        good = group[group["status"] == "ok"]
        per_sketch[str(int(sketch_id))] = {
            "rows": int(len(group)),
            "ok": int(len(good)),
            "max_abs_error": float(group["max_abs_error"].max()),
            "spearman_m2_vs_nano_median": _spearman(good["m2_median_ms"], good["median_ms"]),
            "spearman_excluding_m2_slow_skip": _spearman(
                good.loc[good["m2_status"] == "ok", "m2_median_ms"],
                good.loc[good["m2_status"] == "ok", "median_ms"],
            ),
            "nano_median_ms_range": [float(good["median_ms"].min()), float(good["median_ms"].max())],
        }
    errors = frame[frame["error"].fillna("") != ""]
    wall = frame["wall_s"].to_numpy(dtype=float)
    parts = ("build_time_s", "export_time_s", "upload_time_s", "link_load_time_s", "kernel_wall_s")
    return {
        "selection": {
            "source": "data/pool.parquet (M2, read-only)",
            "seed": REPLAY_SEED,
            "strata": "sketch_id x per-sketch M2 median_ms decile (rank-based qcut)",
            "per_stratum": REPLAY_PER_CELL,
            "rows": int(len(frame)),
            "run_order": "seeded permutation (same seed)",
            "ids": [int(v) for v in frame["id"]],
            "resampled": False,
        },
        "timing_protocol": {
            "repeats": int(frame["repeats"].iloc[0]),
            "number": int(frame["number"].iloc[0]),
            "min_repeat_ms": int(frame["min_repeat_ms"].iloc[0]),
            "source": "configs/workloads/matmul_fp32_512.json (same values as data/run_meta.json M2 config)",
            "timer": "remote time_evaluator on the Nano (harness.time_module)",
            "slow_threshold_during_pilot_ms": "inf",
            "sampler_running": False,
        },
        "status_counts": {k: int(v) for k, v in Counter(frame["status"]).items()},
        "first_errors": errors[["id", "sketch_id", "status", "error"]].head(5).to_dict("records"),
        "correctness": {
            "rtol": 1e-4,
            "atol": 1e-4,
            "checked": int(frame["max_abs_error"].notna().sum()),
            "passed": int((frame["status"].isin(["ok", "slow_skip"])).sum()),
            "max_abs_error_overall": float(frame["max_abs_error"].max()),
        },
        "per_sketch": per_sketch,
        "spearman_pooled_m2_vs_nano_median": _spearman(ok["m2_median_ms"], ok["median_ms"]),
        "spearman_pooled_excluding_m2_slow_skip": _spearman(
            ok.loc[ok["m2_status"] == "ok", "m2_median_ms"],
            ok.loc[ok["m2_status"] == "ok", "median_ms"],
        ),
        "nano_over_m2_median_ratio": {
            "median": float(np.median(ok["median_ms"] / ok["m2_median_ms"])),
            "min": float((ok["median_ms"] / ok["m2_median_ms"]).min()),
            "max": float((ok["median_ms"] / ok["m2_median_ms"]).max()),
        },
        "latency_ms": {
            "min": float(medians.min()),
            "p10": float(np.percentile(medians, 10)),
            "median": pilot_median,
            "p90": float(np.percentile(medians, 90)),
            "max": float(medians.max()),
            "mean": float(medians.mean()),
        },
        "within_schedule_repeat_cv_pct_median": float(
            np.median(100.0 * ok["std_ms"] / ok["median_ms"])
        ),
        "slow_threshold_ms": threshold,
        "slow_threshold_rule": "20 x median of pilot per-schedule medians (status ok)",
        "pilot_rows_above_threshold": int((medians > threshold).sum()),
        "mean_seconds": {part: float(frame[part].mean()) for part in parts}
        | {"wall_s": float(wall.mean())},
        "elapsed_s": float(wall.sum()),
        "device_state_before": summarize_device_state(before),
        "device_state_after": summarize_device_state(after),
        "frequency_residency_during_pilot": frequency_residency(before, after),
    }


def cmd_replay(args: argparse.Namespace) -> int:
    from dvcm.device_state import read_device_state
    from dvcm.harness import measure
    from dvcm.rpc import RPCExecutor

    target, sketches, layouts = nano_design_space()
    replay = select_replay_set(_m2_pool())
    for row in replay.itertuples():
        position = np.asarray(row.vector, dtype=float)
        if not np.array_equal(repair(position, layouts[row.sketch_id]), position):
            raise SystemExit(f"Replay vector for id {row.id} is not a valid Nano position")
    strata = replay.groupby(["sketch_id", "m2_decile"]).size()
    print(f"Replay set: {len(replay)} rows; strata {strata.min()}..{strata.max()} rows each; "
          f"all vectors valid on the Nano layouts (repair is the identity)")
    save_json(QUAL_DIR / "replay_set.json", {"seed": REPLAY_SEED, "rows": replay.to_dict("records")})

    done = pd.read_parquet(REPLAY_TABLE) if REPLAY_TABLE.exists() else pd.DataFrame()
    rows = done.to_dict("records") if not done.empty else []
    measured = {int(r["id"]) for r in rows}
    config = harness_config(target)
    capture = _ErrorCapture()
    logging.getLogger("dvcm.harness").addHandler(capture)
    executor = RPCExecutor.connect(args.host, args.port)
    before = read_device_state(args.ssh_target)
    for index, row in enumerate(replay.itertuples(), start=1):
        if row.id in measured:
            continue
        capture.messages.clear()
        position = np.asarray(row.vector, dtype=float)
        started = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        error = ""
        try:
            result = measure(position, sketches[row.sketch_id], layouts[row.sketch_id], config, executor)
        except Exception as exc:  # RPC/remote failure after a successful build
            result = None
            error = f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:400] if str(exc).strip() else ''}"
            executor = RPCExecutor.connect(args.host, args.port)
        wall = time.perf_counter() - started
        if result is not None:
            timing = result.timing
            setup = result.build_time_s + result.export_time_s + result.upload_time_s + result.link_load_time_s
            error = "; ".join(capture.messages)
        record = {
            "id": int(row.id),
            "sketch_id": int(row.sketch_id),
            "vector": list(row.vector),
            "m2_median_ms": row.m2_median_ms,
            "m2_status": row.m2_status,
            "m2_decile": int(row.m2_decile),
            "run_order": int(row.run_order),
            "status": timing.status if result is not None else "rpc_error",
            "repeats_ms": list(timing.repeats_ms) if result is not None else [],
            "median_ms": timing.median_ms if result is not None else float("nan"),
            "std_ms": timing.std_ms if result is not None else float("nan"),
            "max_abs_error": timing.max_abs_error if result is not None else None,
            "build_time_s": result.build_time_s if result is not None else float("nan"),
            "export_time_s": result.export_time_s if result is not None else float("nan"),
            "upload_time_s": result.upload_time_s if result is not None else float("nan"),
            "link_load_time_s": result.link_load_time_s if result is not None else float("nan"),
            "kernel_wall_s": wall - setup if result is not None else float("nan"),
            "wall_s": wall,
            "repeats": config.repeats,
            "number": config.number,
            "min_repeat_ms": config.min_repeat_ms,
            "timestamp": timestamp,
            "error": error,
        }
        rows.append(record)
        print(f"[{index:3d}/{len(replay)}] id={row.id:<5} sketch={row.sketch_id} dec={row.m2_decile} "
              f"status={record['status']:<12} nano={record['median_ms']:10.3f} ms  "
              f"m2={row.m2_median_ms:8.3f} ms  maxerr={record['max_abs_error']}  wall={wall:6.2f}s",
              flush=True)
        if len(rows) % 10 == 0:
            _checkpoint(rows, REPLAY_TABLE)
    _checkpoint(rows, REPLAY_TABLE)
    after = read_device_state(args.ssh_target)
    frame = pd.DataFrame(rows)
    report = pilot_report(frame, before, after) | {"provenance": provenance()}
    save_json(PILOT_REPORT, report)
    update_run_meta(
        slow_threshold_ms=report["slow_threshold_ms"],
        slow_threshold_source="20 x pilot median (data/nano/pilot_report.json)",
        pilot_report={"path": "data/nano/pilot_report.json", "sha256": _sha256(PILOT_REPORT)},
    )
    print(json.dumps({k: report[k] for k in (
        "status_counts", "first_errors", "correctness", "per_sketch",
        "spearman_pooled_m2_vs_nano_median", "spearman_pooled_excluding_m2_slow_skip",
        "latency_ms", "slow_threshold_ms", "pilot_rows_above_threshold", "mean_seconds",
        "nano_over_m2_median_ratio", "within_schedule_repeat_cv_pct_median")}, indent=2, default=str))
    return 0 if set(report["status_counts"]) == {"ok"} else 1


# --------------------------------------------------------------------------- step 5

CANARY_COUNT = 20
CANARY_SEED = 1
CANARY_PATH = NANO_DATA / "canary.json"
NOISE_TABLE = QUAL_DIR / "noise_rounds.parquet"
NOISE_REPORT = QUAL_DIR / "noise.json"


def select_canaries(pilot: pd.DataFrame, threshold_ms: float) -> pd.DataFrame:
    """20 pilot schedules at evenly spaced latency ranks (collectable range only)."""
    eligible = pilot[(pilot["status"] == "ok") & (pilot["median_ms"] <= threshold_ms)]
    eligible = eligible.sort_values("median_ms").reset_index(drop=True)
    ranks = np.unique(np.round(np.linspace(0, len(eligible) - 1, CANARY_COUNT)).astype(int))
    chosen = eligible.iloc[ranks].copy()
    chosen["latency_rank"] = ranks
    order = np.random.default_rng(CANARY_SEED).permutation(len(chosen))
    chosen = chosen.iloc[order].reset_index(drop=True)
    chosen["canary_id"] = np.arange(len(chosen))
    return chosen


def _busy_estimate_s(median_ms: float, min_repeat_ms: float, repeats: int) -> float:
    """Kernel-busy seconds of one time_module call: correctness run, single shot
    (+ warm-up), evaluator warm-up, then `repeats` repeats of >= min_repeat_ms."""
    return (4.0 * median_ms + repeats * max(median_ms, min_repeat_ms)) / 1000.0


def _load_session(executor: Any, canaries: pd.DataFrame, sketches: list, layouts: list, config: HarnessConfig) -> tuple[list, list[dict]]:
    from dvcm.harness import build_from_position

    modules, setup = [], []
    for row in canaries.itertuples():
        started = time.perf_counter()
        library, status = build_from_position(
            np.asarray(row.vector, dtype=float), sketches[row.sketch_id], layouts[row.sketch_id], config
        )
        build_s = time.perf_counter() - started
        if library is None:
            raise RuntimeError(f"canary id {row.id} failed to build ({status})")
        remote = executor.load_module(library)
        modules.append(remote.module)
        setup.append({"id": int(row.id), "build_time_s": build_s, "export_time_s": remote.export_time_s,
                      "upload_time_s": remote.upload_time_s, "link_load_time_s": remote.link_load_time_s})
    return modules, setup


def _probe(ssh_target: str) -> tuple[dict, float]:
    from dvcm.device_state import read_device_state

    started = time.perf_counter()
    state = read_device_state(ssh_target)
    return state, time.perf_counter() - started


def _timed_round(executor: Any, modules: list, canaries: pd.DataFrame, config: HarnessConfig,
                 session: int, round_id: int, ssh_target: str) -> tuple[list[dict], dict]:
    from dvcm.device_state import frequency_residency, summarize_device_state
    from dvcm.harness import time_module

    before, before_s = _probe(ssh_target)
    start_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    rows = []
    for module, row in zip(modules, canaries.itertuples()):
        timing = time_module(module, config, executor)
        rows.append({
            "session": session, "round": round_id, "canary_id": int(row.canary_id), "id": int(row.id),
            "sketch_id": int(row.sketch_id), "pilot_median_ms": float(row.median_ms), "status": timing.status,
            "repeats_ms": list(timing.repeats_ms), "median_ms": timing.median_ms, "std_ms": timing.std_ms,
            "max_abs_error": timing.max_abs_error,
        })
    ended = time.monotonic()
    after, after_s = _probe(ssh_target)
    busy = sum(_busy_estimate_s(r["median_ms"], config.min_repeat_ms, config.repeats) for r in rows)
    residency = frequency_residency(before, after)
    record = {
        "session": session, "round": round_id, "start_utc": start_utc, "timing_wall_s": ended - started,
        "kernel_busy_estimate_s": busy, "probe_seconds": [before_s, after_s],
        "before": summarize_device_state(before), "after": summarize_device_state(after),
        "frequency_residency": residency, "_ended_monotonic": ended,
    }
    policy = residency.get("policy0", {})
    print(f"session {session} round {round_id}: timing {ended - started:6.1f}s busy~{busy:5.1f}s "
          f"at-max {policy.get('seconds_at_max', float('nan')):6.2f}s "
          f"window {policy.get('total_s', float('nan')):6.1f}s "
          f"CPU-therm {record['before']['thermal_millic'].get('CPU-therm')}->{record['after']['thermal_millic'].get('CPU-therm')} "
          f"statuses {dict(Counter(r['status'] for r in rows))}", flush=True)
    return rows, record


def noise_report(rows: list[dict], rounds: list[dict], canaries: pd.DataFrame) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    first = frame[frame["session"] == 1].pivot(index="canary_id", columns="round", values="median_ms")
    within_cv = (
        frame[frame["session"] == 1]
        .assign(cv=lambda f: 100.0 * f["std_ms"] / f["median_ms"])
        .groupby("canary_id")["cv"].median()
    )
    grand = first.median(axis=1)
    per_canary = []
    for canary_id, values in first.iterrows():
        values = values.to_numpy(dtype=float)
        meta = canaries.set_index("canary_id").loc[canary_id]
        per_canary.append({
            "canary_id": int(canary_id), "id": int(meta["id"]), "sketch_id": int(meta["sketch_id"]),
            "pilot_median_ms": float(meta["median_ms"]), "round_medians_ms": values.tolist(),
            "cv_across_rounds_pct": float(100.0 * values.std() / values.mean()),
            "max_over_min": float(values.max() / values.min()),
            "within_round_repeat_cv_pct_median": float(within_cv.loc[canary_id]),
        })
    drift = []
    for round_id in first.columns:
        drift.append({
            "round": int(round_id),
            "median_rel_vs_round0_pct": float(100.0 * np.median(first[round_id] / first[0] - 1.0)),
            "median_rel_vs_canary_median_pct": float(100.0 * np.median(first[round_id] / grand - 1.0)),
        })
    second = frame[frame["session"] == 2].set_index("canary_id")["median_ms"]
    session_rel = (second / grand.loc[second.index] - 1.0) * 100.0
    session_vs_last = (second / first[first.columns.max()].loc[second.index] - 1.0) * 100.0
    frequency = []
    for record in rounds:
        policy = record["frequency_residency"].get("policy0", {})
        below = {k: v for k, v in policy.get("seconds_by_khz", {}).items()
                 if int(k) < (policy.get("max_khz") or 0) and v > 0}
        frequency.append({
            "session": record["session"], "round": record["round"],
            "timing_wall_s": record["timing_wall_s"], "kernel_busy_estimate_s": record["kernel_busy_estimate_s"],
            "window_s": policy.get("total_s"), "seconds_at_max": policy.get("seconds_at_max"),
            "seconds_below_max": float(sum(below.values())),
            "at_max_covers_busy": (policy.get("seconds_at_max") or 0) >= 0.95 * record["kernel_busy_estimate_s"],
            "scaling_max_khz_after": record["after"]["max_khz"],
            "cpu_therm_millic": [record["before"]["thermal_millic"].get("CPU-therm"),
                                 record["after"]["thermal_millic"].get("CPU-therm")],
            "active_cooling_states_after": record["after"]["active_cooling_states"],
            "governors": record["after"]["governors"], "nvpmodel": record["after"]["nvpmodel"],
            "loadavg_before": record["before"]["loadavg"], "loadavg_after": record["after"]["loadavg"],
        })
    starts = [datetime.fromisoformat(r["start_utc"]) for r in rounds]
    return {
        "canaries": canaries[["canary_id", "id", "sketch_id", "median_ms", "latency_rank"]]
        .rename(columns={"median_ms": "pilot_median_ms"}).to_dict("records"),
        "round_start_gaps_s": [(b - a).total_seconds() for a, b in zip(starts, starts[1:])],
        "per_canary": per_canary,
        "cv_across_rounds_pct": {
            "median": float(np.median([c["cv_across_rounds_pct"] for c in per_canary])),
            "max": float(np.max([c["cv_across_rounds_pct"] for c in per_canary])),
        },
        "median_drift_per_round": drift,
        "second_session": {
            "median_rel_diff_vs_session1_canary_median_pct": float(np.median(session_rel)),
            "median_abs_rel_diff_pct": float(np.median(np.abs(session_rel))),
            "max_abs_rel_diff_pct": float(np.max(np.abs(session_rel))),
            "median_rel_diff_vs_session1_last_round_pct": float(np.median(session_vs_last)),
            "per_canary_rel_diff_pct": {int(k): float(v) for k, v in session_rel.items()},
        },
        "frequency": frequency,
        "statuses": {k: int(v) for k, v in Counter(frame["status"]).items()},
    }


def cmd_noise(args: argparse.Namespace) -> int:
    import gc

    from dvcm.rpc import RPCExecutor

    meta = load_run_meta()
    threshold = float(meta["slow_threshold_ms"])
    prov = provenance()
    canaries = select_canaries(pd.read_parquet(REPLAY_TABLE), threshold)
    save_json(CANARY_PATH, [
        {"canary_id": int(r.canary_id), "id": int(r.id), "sketch_id": int(r.sketch_id),
         "vector": [int(v) for v in r.vector], "pilot_median_ms": float(r.median_ms)}
        for r in canaries.itertuples()
    ])
    print(f"Canaries ({len(canaries)}): pilot medians "
          f"{sorted(round(float(v), 2) for v in canaries['median_ms'])} ms", flush=True)
    target, sketches, layouts = nano_design_space()
    config = harness_config(target, threshold)
    rows: list[dict] = []
    rounds: list[dict] = []
    executor = RPCExecutor.connect(args.host, args.port)
    modules, setup_first = _load_session(executor, canaries, sketches, layouts, config)
    last_end = None
    for round_id in range(args.rounds):
        if last_end is not None:
            time.sleep(max(0.0, last_end + args.gap_s - time.monotonic()))
        round_rows, record = _timed_round(executor, modules, canaries, config, 1, round_id, args.ssh_target)
        last_end = record.pop("_ended_monotonic")
        rows.extend(round_rows)
        rounds.append(record)
        _checkpoint(rows, NOISE_TABLE)
        save_json(QUAL_DIR / "noise_rounds_meta.json", rounds)
    del modules, executor
    gc.collect()
    time.sleep(max(0.0, last_end + args.gap_s - time.monotonic()))
    executor = RPCExecutor.connect(args.host, args.port)
    modules, setup_second = _load_session(executor, canaries, sketches, layouts, config)
    round_rows, record = _timed_round(executor, modules, canaries, config, 2, 0, args.ssh_target)
    record.pop("_ended_monotonic")
    rows.extend(round_rows)
    rounds.append(record)
    _checkpoint(rows, NOISE_TABLE)
    report = noise_report(rows, rounds, canaries) | {
        "protocol": {
            "rounds": args.rounds, "min_gap_between_rounds_s": args.gap_s,
            "gap_definition": "end of one round's timing to start of the next round's timing (probe excluded)",
            "timing": "harness.time_module per canary (collection protocol; modules loaded once per session)",
            "slow_threshold_ms": threshold, "sampler_running_during_timing": False,
            "device_state": "read-only probe before and after each round only",
        },
        "rounds": rounds,
        "setup": {"session1": setup_first, "session2": setup_second},
        "provenance": prov,
    }
    save_json(NOISE_REPORT, report)
    print(json.dumps({k: report[k] for k in ("round_start_gaps_s", "cv_across_rounds_pct",
                                              "median_drift_per_round", "second_session", "statuses")},
                     indent=2))
    return 0


# --------------------------------------------------------------------------- step 6

PROJECTION_SIZES = (752, 1500, 2500, 3500, 5000)


def cmd_throughput(_: argparse.Namespace) -> int:
    import runpy

    collector = runpy.run_path(str(ROOT / "scripts" / "01_collect.py"), run_name="throughput")
    batch_size, cooldown_s = int(collector["BATCH_SIZE"]), float(collector["COOLDOWN_S"])
    threshold = float(load_run_meta()["slow_threshold_ms"])
    pilot = pd.read_parquet(REPLAY_TABLE)
    ok = pilot[pilot["status"] == "ok"].copy()
    noise = json.loads(NOISE_REPORT.read_text(encoding="utf-8"))
    canary_ids = [int(item["id"]) for item in noise["canaries"]]
    probe_s = float(np.mean([s for record in noise["rounds"] for s in record["probe_seconds"]]))

    busy = np.array([_busy_estimate_s(m, r, k) for m, r, k in
                     zip(ok["median_ms"], ok["min_repeat_ms"], ok["repeats"])])
    overhead = ok["kernel_wall_s"].to_numpy() - busy
    slow = ok["median_ms"].to_numpy() > threshold
    # Above the threshold the collector stops after the correctness run and single shot.
    kernel = np.where(slow, np.median(overhead) + 3.0 * ok["median_ms"].to_numpy() / 1000.0,
                      ok["kernel_wall_s"].to_numpy())
    phases = {
        "build_s": float(ok["build_time_s"].mean()),
        "export_s": float(ok["export_time_s"].mean()),
        "upload_s": float(ok["upload_time_s"].mean()),
        "link_load_s": float(ok["link_load_time_s"].mean()),
        "kernel_timing_s": float(kernel.mean()),
    }
    schedule_s = sum(phases.values())
    canary_rows = pilot[pilot["id"].isin(canary_ids)]
    canary_batch_s = float(canary_rows["wall_s"].sum())
    projections = []
    for n in PROJECTION_SIZES:
        batches = math.ceil(n / batch_size)
        parts = {
            "schedules_s": n * schedule_s,
            "canaries_s": batches * canary_batch_s,
            "device_probes_s": batches * 2 * probe_s,
            "cooldown_s": (batches - 1) * cooldown_s,
        }
        total = sum(parts.values())
        projections.append({"n": n, "batches": batches, **parts, "total_s": total,
                            "total_h": total / 3600.0, "seconds_per_schedule": total / n})
    at_full = projections[-1]
    per_schedule = {**phases,
                    "cooldown_amortised_s": at_full["cooldown_s"] / at_full["n"],
                    "canaries_amortised_s": at_full["canaries_s"] / at_full["n"],
                    "device_probes_amortised_s": at_full["device_probes_s"] / at_full["n"]}
    print(f"Planned batch size {batch_size}, cooldown {cooldown_s:.0f} s/batch, "
          f"{len(canary_ids)} canaries/batch ({canary_batch_s:.1f} s), "
          f"2 device probes/batch ({probe_s:.1f} s each); slow threshold {threshold:.1f} ms "
          f"({int(slow.sum())}/{len(ok)} pilot rows above)")
    print("Mean seconds per schedule (pilot, n=%d):" % len(ok))
    for name, value in per_schedule.items():
        print(f"  {name:<28} {value:8.3f}")
    print(f"  {'schedule (build..kernel)':<28} {schedule_s:8.3f}")
    print(f"  {'all-in at 5,000':<28} {at_full['seconds_per_schedule']:8.3f}")
    print(f"{'N':>6} {'batches':>7} {'schedules h':>11} {'canaries h':>10} {'probes h':>8} "
          f"{'cooldown h':>10} {'total h':>8}")
    for p in projections:
        print(f"{p['n']:>6,} {p['batches']:>7} {p['schedules_s'] / 3600:>11.2f} "
              f"{p['canaries_s'] / 3600:>10.2f} {p['device_probes_s'] / 3600:>8.2f} "
              f"{p['cooldown_s'] / 3600:>10.2f} {p['total_h']:>8.2f}")
    save_json(QUAL_DIR / "throughput.json", {
        "batch_size": batch_size, "cooldown_s": cooldown_s, "canaries_per_batch": len(canary_ids),
        "canary_batch_s": canary_batch_s, "probe_s": probe_s, "slow_threshold_ms": threshold,
        "pilot_rows": len(ok), "pilot_rows_above_threshold": int(slow.sum()),
        "per_schedule_s": per_schedule, "schedule_s": schedule_s,
        "kernel_rpc_overhead_s_median": float(np.median(overhead)),
        "projections": projections, "provenance": provenance(),
    })
    return 0


# --------------------------------------------------------------------------- step 8

M2_DEFAULTS = {"l1_bytes": 131072, "l2_bytes": 16777216, "peak_flops": 4e11, "bandwidth_bytes_s": 1e11}


def _profile_lines(path: Path, keys: tuple[str, ...]) -> dict[str, list[int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {key: [n for n, line in enumerate(lines, 1) if f'"{key}"' in line] for key in keys}


def cmd_features(_: argparse.Namespace) -> int:
    from dvcm.analytic import AnalyticConstants, latency_seconds, penalty_terms
    from dvcm.features import ALL_DERIVED_NAMES, derived, features
    from dvcm.study import load_layouts

    nano = load_target_config("jetson_nano", ROOT)
    constants = AnalyticConstants.from_target(nano)
    m2_constants = AnalyticConstants.from_target(load_target_config("apple_m2", ROOT))
    leaked = {k: v for k, v in M2_DEFAULTS.items() if getattr(constants, k) == v}
    profile_path = ROOT / "configs" / "hardware" / "jetson_nano.json"
    where = _profile_lines(profile_path, ("peak_flops", "bandwidth_bytes_s", "l1_data_cache_bytes",
                                          "l2_cache_bytes", "simd_width_f32", "num_cores", "vector_registers"))
    print(f"Nano AnalyticConstants.from_target(): {constants.to_dict()}")
    for key in ("peak_flops", "bandwidth_bytes_s"):
        print(f"  {key} = {getattr(constants, key):.6g}  [{profile_path.relative_to(ROOT)} lines {where[key]}] "
              f"note: {nano.notes[key][:60]}...")
    print(f"M2 default values present on the Nano path: {leaked or 'none'}")
    layouts = load_layouts(NANO_DATA, nano)
    bound = {sid: (l["vector_lanes"], l["num_cores"], l["vector_registers"]) for sid, l in layouts.items()}
    print(f"Feature binding from data/nano/layout_*.json + Nano profile (lanes, cores, registers): {bound}")

    replay = pd.DataFrame(json.loads((QUAL_DIR / "replay_set.json").read_text())["rows"])
    expected_flops = 2.0 * 512**3
    problems, records = [], []
    for row in replay.itertuples():
        position = np.asarray(row.vector, dtype=float)
        layout = layouts[row.sketch_id]
        vector = features(position, row.sketch_id, layout)
        descriptors = derived(position, layout)
        penalties = penalty_terms(position, layout, constants)
        latency = latency_seconds(position, layout, constants)
        levels = [descriptors[f"tile_level_{level}_bytes"] for level in range(4)]
        checks = {
            "features_finite": bool(np.isfinite(vector).all()),
            "descriptors_finite": all(math.isfinite(descriptors[n]) for n in ALL_DERIVED_NAMES),
            "penalties_in_(0,1]": all(math.isfinite(v) and 0.0 < v <= 1.0 for v in penalties.values()),
            "utilisations_in_(0,1]": all(0.0 < descriptors[n] <= 1.0 for n in (
                "vector_lane_utilisation", "parallel_utilisation", "register_utilisation")),
            "flop_invariant": descriptors["flop_count"] == expected_flops,
            "footprints_positive_nonincreasing": levels[3] > 0 and all(a >= b for a, b in zip(levels, levels[1:])),
            "level0_is_whole_problem": levels[0] == 3 * 4 * 512**2,
            "bytes_moved_ge_level1": descriptors["bytes_moved"] >= levels[1] > 0,
            "latency_positive_finite": math.isfinite(latency) and latency > 0,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            problems.append({"id": int(row.id), "failed": failed})
        records.append({"id": int(row.id), "sketch_id": int(row.sketch_id), **descriptors,
                        **{f"penalty_{k}": v for k, v in penalties.items()}, "analytic_latency_s": latency,
                        "l2_spill": levels[1] > constants.l2_bytes, "l1_spill": levels[2] > constants.l1_bytes,
                        "l2_spill_with_m2_l2": levels[1] > m2_constants.l2_bytes})
    frame = pd.DataFrame(records)
    ranges = {name: [float(frame[name].min()), float(frame[name].max())]
              for name in list(ALL_DERIVED_NAMES) + [c for c in frame if c.startswith("penalty_")]}
    summary = {
        "rows": len(frame),
        "checks_failed": problems,
        "flop_invariant_value": expected_flops,
        "l2_bytes": constants.l2_bytes,
        "l2_spill_fraction": float(frame["l2_spill"].mean()),
        "l2_spill_count": int(frame["l2_spill"].sum()),
        "l2_spill_fraction_by_sketch": frame.groupby("sketch_id")["l2_spill"].mean().round(4).to_dict(),
        "l1_spill_fraction_level2_vs_l1": float(frame["l1_spill"].mean()),
        "l2_spill_fraction_if_m2_l2_16MiB": float(frame["l2_spill_with_m2_l2"].mean()),
        "cache_l2_penalty_below_1_fraction": float((frame["penalty_cache_l2"] < 1.0).mean()),
        "ranges": ranges,
        "m2_defaults_on_nano_path": leaked,
        "constants": constants.to_dict(),
        "constant_sources": {"file": str(profile_path.relative_to(ROOT)), "lines": where,
                             "peak_flops_note": nano.notes["peak_flops"],
                             "bandwidth_bytes_s_note": nano.notes["bandwidth_bytes_s"]},
        "provenance": provenance(),
    }
    save_json(QUAL_DIR / "nano_features.json", summary)
    frame.to_parquet(QUAL_DIR / "nano_features.parquet", index=False)
    print(json.dumps({k: summary[k] for k in (
        "rows", "checks_failed", "l2_spill_fraction", "l2_spill_count", "l2_spill_fraction_by_sketch",
        "l1_spill_fraction_level2_vs_l1", "l2_spill_fraction_if_m2_l2_16MiB",
        "cache_l2_penalty_below_1_fraction", "ranges")}, indent=1))
    return 1 if problems or leaked else 0


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("parity")
    loop_order = sub.add_parser("loop-order")
    loop_order.add_argument("--print-tir", action="store_true")
    for name in ("replay", "noise"):
        rpc = sub.add_parser(name)
        rpc.add_argument("--host", default="127.0.0.1")
        rpc.add_argument("--port", type=int, default=9090)
        rpc.add_argument("--ssh-target", required=True)
        if name == "noise":
            rpc.add_argument("--rounds", type=int, default=5)
            rpc.add_argument("--gap-s", type=float, default=300.0)
    sub.add_parser("features")
    sub.add_parser("throughput")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    commands = {"parity": cmd_parity, "loop-order": cmd_loop_order, "replay": cmd_replay,
                "noise": cmd_noise, "features": cmd_features, "throughput": cmd_throughput}
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
