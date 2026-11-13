"""scripts/report_nano_collection.py on a synthetic collection directory."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _reporter():
    spec = importlib.util.spec_from_file_location("report_test", ROOT / "scripts" / "report_nano_collection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collection(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "nano"
    data.mkdir()
    rng = np.random.default_rng(0)
    m2_ms = rng.lognormal(1.0, 0.5, 40)
    state = json.dumps({"before": {"active_cooling_states": {"CORE-floor": 1}},
                        "after": {"active_cooling_states": {"CORE-floor": 1}},
                        "rpc_session_index": {"batch_start": 0, "batch_end": 0}})
    statuses = ["ok"] * 39 + ["slow_skip"]
    pd.DataFrame({
        "id": range(40), "sketch_id": [i % 2 for i in range(40)], "vector": [[0] * 11] * 40,
        "median_ms": m2_ms * 12.0, "status": statuses, "batch_id": [i // 20 for i in range(40)],
        "device_state": [state] * 40,
    }).to_parquet(data / "pool.parquet", index=False)
    m2 = tmp_path / "m2_pool.parquet"
    pd.DataFrame({"id": range(40), "sketch_id": [i % 2 for i in range(40)],
                  "median_ms": m2_ms, "status": ["ok"] * 40}).to_parquet(m2, index=False)
    batch = {"batch_id": 0, "ended_utc": "2026-11-13T23:00:00+00:00", "rows": 20,
             "status_counts": {"ok": 20}, "canary_drift_pct": 0.4, "canary_ok": 20,
             "frequency": {"mean_mhz": 1100.0, "seconds_at_max": 200.0, "busy_estimate_s": 150.0,
                           "drop": False, "reasons": []},
             "temperature_c": {"min": 29.5, "max": 34.0}, "pause_reason": None}
    (data / "collection_batches.jsonl").write_text(
        json.dumps(batch) + "\n" + json.dumps({**batch, "batch_id": 1}) + "\n"
    )
    (data / "run_meta.json").write_text(json.dumps(
        {"collection": {"state": "running", "reconnects": 0, "rpc_sessions_opened": 1,
                        "last_update": "2026-11-13T23:00:01+00:00"}}
    ))
    return data, m2


def test_report_covers_required_sections_and_is_idempotent(tmp_path):
    reporter = _reporter()
    data, m2 = _collection(tmp_path)
    first = reporter.build_report(data, m2)
    assert first == reporter.build_report(data, m2)  # no wall-clock content
    assert "slow_skip fraction: 0.0250 (1/40)" in first
    assert "Stayed in one RPC session: **yes**" in first
    assert "| pooled | 39 | 1 |" in first  # Nano = 12 x M2 in this fixture
    assert "min 29.5 C, max 34.0 C" in first
    assert "CPU cap/throttle cooling devices seen active: none" in first
    assert reporter._sha256(data / "pool.parquet") in first
    assert "| 39 |" in first.split("## Nano latency distribution")[1]
