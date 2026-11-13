#!/usr/bin/env python3
"""Write data/nano/collection_report.md from the Nano collection outputs.

Read-only on its inputs and idempotent: the report depends only on the files it
reads (it is stamped with the collection's own last_update, not the wall clock),
so it can be rerun at any time, including while the collection is running.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CPU_CAP_DEVICES = (
    "cpu-balanced", "cpu-throttle-alert", "DFLL-cap", "CORE-cap",
    "emergency-balanced", "heavy", "hot-surface-alert", "skin-balanced",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _spearman(x: Any, y: Any) -> float | None:
    from scipy.stats import spearmanr

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    return float(spearmanr(x, y).statistic)


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(data_dir: Path, m2_pool_path: Path) -> str:
    pool_path = data_dir / "pool.parquet"
    meta_path = data_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    collection = meta.get("collection", {})
    lines = [
        "# Nano collection report",
        "",
        f"Data as of {collection.get('last_update', 'n/a')} (collection `last_update`), from "
        f"`{data_dir}`. Regenerate with `python scripts/report_nano_collection.py`.",
        "",
    ]
    if not pool_path.is_file():
        return "\n".join(lines + ["No `pool.parquet` yet.", ""])
    pool = pd.read_parquet(pool_path)
    batches = _read_jsonl(data_dir / "collection_batches.jsonl")

    lines += ["## Run", ""]
    lines += _table(
        ["field", "value"],
        [[name, collection.get(name)] for name in (
            "state", "pause_reason", "git_commit", "n_target", "measured", "batch_size",
            "cooldown_s", "canaries", "canary_baseline", "slow_threshold_ms", "reconnects",
            "start_timestamp", "last_update",
        )],
    )

    counts = Counter(pool["status"])
    lines += ["", "## Rows by status", ""]
    lines += _table(
        ["status", "rows", "fraction"],
        [[name, count, count / len(pool)] for name, count in sorted(counts.items())]
        + [["total", len(pool), 1.0]],
    )
    lines += ["", f"slow_skip fraction: {counts.get('slow_skip', 0) / len(pool):.4f} "
              f"({counts.get('slow_skip', 0)}/{len(pool)})"]

    sessions = pool.groupby("batch_id")["device_state"].first().map(
        lambda text: json.loads(text).get("rpc_session_index", {})
    )
    session_rows = [[int(batch), s.get("batch_start"), s.get("batch_end")] for batch, s in sessions.items()]
    indices = {row[1] for row in session_rows} | {row[2] for row in session_rows}
    opened = collection.get("rpc_sessions_opened")
    one_session = len(indices) <= 1 and (opened == 1 if opened is not None else not collection.get("reconnects"))
    lines += ["", "## RPC sessions", "",
              f"Stayed in one RPC session: **{'yes' if one_session else 'no'}** "
              f"(RPC sessions opened: {opened if opened is not None else 'n/a'}; session indices "
              f"seen: {sorted(indices)}; reconnects in the latest invocation: "
              f"{collection.get('reconnects', 'n/a')}).", ""]
    lines += _table(["batch", "session index at start", "session index at end"], session_rows)

    lines += ["", "## Canary drift, frequency and temperature per batch", ""]
    if batches:
        lines += _table(
            ["batch", "ended (UTC)", "rows", "statuses", "canary drift %", "canaries ok",
             "mean MHz", "s at max", "busy s", "freq drop", "min C", "max C", "pause"],
            [[b["batch_id"], b["ended_utc"][:19], b["rows"], b["status_counts"],
              b["canary_drift_pct"], b["canary_ok"], b["frequency"].get("mean_mhz"),
              b["frequency"].get("seconds_at_max"), b["frequency"].get("busy_estimate_s"),
              b["frequency"].get("drop"), b["temperature_c"]["min"], b["temperature_c"]["max"],
              b.get("pause_reason") or ""] for b in batches],
        )
    else:
        lines += ["No batch log (`collection_batches.jsonl`) yet."]

    temperatures = [t for b in batches for t in (b["temperature_c"]["min"], b["temperature_c"]["max"])
                    if t is not None]
    drops = [(b["batch_id"], b["frequency"].get("reasons")) for b in batches if b["frequency"].get("drop")]
    caps = Counter()
    for text in pool["device_state"].unique():
        state = json.loads(text)
        for side in ("before", "after"):
            for name, value in ((state.get(side) or {}).get("active_cooling_states") or {}).items():
                if name in CPU_CAP_DEVICES and value:
                    caps[name] += 1
    lines += ["", "## Temperature and throttling", "",
              f"Temperature over all batch probes (PMIC-Die placeholder excluded): min "
              f"{min(temperatures, default=float('nan')):.1f} C, max "
              f"{max(temperatures, default=float('nan')):.1f} C.",
              f"Frequency-drop batches: {drops or 'none'}.",
              f"CPU cap/throttle cooling devices seen active: {dict(caps) or 'none'}."]

    lines += ["", "## Spearman: M2 vs Nano medians (measured ids, Nano status ok)", ""]
    if m2_pool_path.is_file():
        m2 = pd.read_parquet(m2_pool_path, columns=["id", "sketch_id", "median_ms", "status"])
        paired = pool[pool["status"] == "ok"][["id", "sketch_id", "median_ms"]].merge(
            m2.rename(columns={"median_ms": "m2_median_ms", "status": "m2_status"}),
            on=["id", "sketch_id"],
        )
        paired = paired[np.isfinite(paired["median_ms"]) & np.isfinite(paired["m2_median_ms"])]
        rows = [["pooled", len(paired), _spearman(paired["m2_median_ms"], paired["median_ms"])]]
        rows += [[f"sketch {int(s)}", len(g), _spearman(g["m2_median_ms"], g["median_ms"])]
                 for s, g in paired.groupby("sketch_id")]
        lines += _table(["set", "n", "Spearman"], rows)
    else:
        lines += [f"M2 pool `{m2_pool_path}` not found."]

    ok = pool.loc[pool["status"] == "ok", "median_ms"].to_numpy(dtype=float)
    lines += ["", "## Nano latency distribution (status ok, per-schedule median)", ""]
    if ok.size:
        q = np.percentile(ok, [0, 10, 25, 50, 75, 90, 100])
        lines += _table(["n", "min ms", "p10", "p25", "median", "p75", "p90", "max ms", "mean ms"],
                        [[int(ok.size), *map(float, q), float(ok.mean())]])
    else:
        lines += ["No ok rows yet."]

    lines += ["", "## Outputs", ""]
    outputs = [pool_path, data_dir / "canary_log.parquet", data_dir / "collection_batches.jsonl",
               data_dir / "collection_errors.jsonl", meta_path]
    lines += _table(["path", "sha256"], [[f"`{p}`", _sha256(p) if p.is_file() else "absent"]
                                         for p in outputs])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "nano")
    parser.add_argument("--m2-pool", type=Path, default=ROOT / "data" / "pool.parquet")
    parser.add_argument("--output", type=Path, help="default: <data-dir>/collection_report.md")
    args = parser.parse_args()
    output = args.output or args.data_dir / "collection_report.md"
    report = build_report(args.data_dir, args.m2_pool)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
