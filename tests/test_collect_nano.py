"""Positions-driven (Nano) collection path in scripts/01_collect.py; no device is used."""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from dvcm.config import load_target_config, load_workload_config
from dvcm.encoding import describe_space, layout_to_json, random_position
from dvcm.harness import MeasurementResult, TimingResult

ROOT = Path(__file__).resolve().parents[1]
SERVER_ENV = {
    "TVM_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "TVM_BIND_THREADS": "1",
    "TVM_EXCLUDE_WORKER0": "1",
    "TVM_THREAD_POOL_SPIN_COUNT": "0",
}


def _collector():
    spec = importlib.util.spec_from_file_location(
        "collector_positions_test", ROOT / "scripts" / "01_collect.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _qualified_output(collector, out: Path, canaries=None) -> None:
    _write_json(out / "pilot_report.json", {"slow_threshold_ms": 700.0})
    _write_json(
        out / "run_meta.json",
        {
            "slow_threshold_ms": 700.0,
            "pilot_report": {"sha256": collector._sha256(out / "pilot_report.json")},
            "rpc_server": {"environment": {**SERVER_ENV, "CC": "/usr/bin/gcc"}},
        },
    )
    if canaries is not None:
        _write_json(out / "canary.json", canaries)


def test_test_first_order_is_test_then_nested_train_then_validation():
    collector = _collector()
    splits = {
        "test": [9, 3],
        "validation": [7, 1],
        "train": [2, 4, 5, 6, 8],
        "nested_train": {"2": [5, 2], "4": [5, 2, 8, 4], "all": [5, 2, 8, 4, 6]},
    }
    assert collector.order_test_first(splits, set(range(1, 10))) == [9, 3, 5, 2, 8, 4, 6, 7, 1]
    with pytest.raises(ValueError, match="missing from the splits"):
        collector.order_test_first(splits, set(range(1, 11)))
    with pytest.raises(ValueError, match="missing from the source pool"):
        collector.order_test_first(splits, set(range(2, 10)))


def test_test_first_order_on_m2_splits_when_present():
    path = ROOT / "data" / "splits.json"
    if not path.is_file():
        pytest.skip("M2 splits.json not present")
    splits = json.loads(path.read_text(encoding="utf-8"))
    ids = set(splits["train"]) | set(splits["validation"]) | set(splits["test"])
    order = _collector().order_test_first(splits, ids)
    n_test = len(splits["test"])
    assert len(order) == len(ids) == len(set(order))
    assert order[:n_test] == splits["test"]
    assert order[n_test : n_test + 100] == splits["nested_train"]["100"]
    train_end = n_test + len(splits["train"])
    assert order[n_test:train_end] == splits["nested_train"]["all"]
    assert set(order[train_end:]) == set(splits["validation"])


def test_collection_refuses_without_threshold_and_pilot_report(tmp_path):
    collector = _collector()
    with pytest.raises(SystemExit, match="does not exist"):
        collector.require_pilot_threshold(tmp_path)
    _write_json(tmp_path / "run_meta.json", {"rpc_server": {"environment": SERVER_ENV}})
    with pytest.raises(SystemExit, match="slow_threshold_ms"):
        collector.require_pilot_threshold(tmp_path)
    meta = {"rpc_server": {"environment": SERVER_ENV}, "slow_threshold_ms": 700.0}
    _write_json(tmp_path / "run_meta.json", meta)
    with pytest.raises(SystemExit, match="pilot report"):
        collector.require_pilot_threshold(tmp_path)
    _write_json(tmp_path / "pilot_report.json", {"slow_threshold_ms": 650.0})
    with pytest.raises(SystemExit, match="does not match"):
        collector.require_pilot_threshold(tmp_path)
    _write_json(tmp_path / "pilot_report.json", {"slow_threshold_ms": 700.0})
    with pytest.raises(SystemExit, match="checksum"):
        collector.require_pilot_threshold(tmp_path)
    meta["pilot_report"] = {"sha256": collector._sha256(tmp_path / "pilot_report.json")}
    _write_json(tmp_path / "run_meta.json", meta)
    assert collector.require_pilot_threshold(tmp_path)["slow_threshold_ms"] == 700.0


def test_nano_positions_path_stays_gated_and_never_connects(tmp_path, monkeypatch):
    collector = _collector()
    _qualified_output(collector, tmp_path)
    monkeypatch.setattr(
        collector.RPCExecutor, "connect", lambda *_: pytest.fail("must not connect")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "01_collect.py", "--target", "jetson_nano",
            "--positions-from", str(tmp_path / "source" / "pool.parquet"),
            "--order", "test_first", "--rpc-host", "127.0.0.1",
            "--ssh-target", "nano", "--output-dir", str(tmp_path),
        ],
    )
    with pytest.raises(RuntimeError, match="not configured"):
        collector.main()
    assert not (tmp_path / "pool.parquet").exists()


def test_positions_collection_checkpoints_per_batch_resumes_and_retries_once(
    tmp_path, monkeypatch
):
    collector = _collector()
    profile = replace(
        load_target_config("jetson_nano"), schedule_rules="llvm", postprocessors="llvm"
    )
    workload = load_workload_config()
    context = collector.ms.TuneContext(
        mod=collector.make_matmul(workload.n),
        target=profile.make_tvm_target(),
        space_generator=collector.PostOrderApply(sch_rules="llvm", postprocs="llvm"),
        rand_state=workload.seed,
        num_threads=4,
    )
    layouts = [describe_space(sketch.trace) for sketch in context.generate_design_space()]
    source, out = tmp_path / "m2", tmp_path / "nano"
    source.mkdir()
    out.mkdir()
    for sketch_id, layout in enumerate(layouts):
        _write_json(source / f"layout_{sketch_id}.json", layout_to_json(layout, sketch_id))
    rng = np.random.default_rng(3)
    ids = [10, 11, 12, 13, 14]
    vectors = {i: random_position(rng, layouts[i % 3]).astype(int).tolist() for i in ids}
    pd.DataFrame(
        {"id": ids, "sketch_id": [i % 3 for i in ids], "vector": [vectors[i] for i in ids],
         "median_ms": [1.0] * 5, "status": ["ok"] * 5}
    ).to_parquet(source / "pool.parquet", index=False)
    _write_json(
        source / "splits.json",
        {"test": [12, 10], "validation": [], "train": [11, 13, 14],
         "nested_train": {"2": [13, 11], "all": [13, 11, 14]}},
    )
    _qualified_output(
        collector, out, canaries=[{"sketch_id": 12 % 3, "vector": vectors[12], "id": 12}]
    )

    by_vector = {tuple(v): i for i, v in vectors.items()}
    failures = {11: 1, 14: 2}
    calls, connects, sleeps = [], [], []

    def fake_measure(position, sketch, layout, config, executor):
        assert config.slow_threshold_ms == 700.0
        sample_id = by_vector[tuple(int(v) for v in position)]
        calls.append(sample_id)
        if failures.get(sample_id, 0):
            failures[sample_id] -= 1
            raise ConnectionResetError("RPC session dropped")
        return MeasurementResult(TimingResult((1.0, 1.2), 1.1, 0.1, "ok", 0.0), 0.1, 0.2, 0.3, 0.4)

    state = {"rpc_servers": [{"pid": 1, "args": "rpc_server", "environment": dict(SERVER_ENV)}]}
    executor = type("Executor", (), {"input_prepare_time_s": 0.0, "input_upload_time_s": 0.0})
    monkeypatch.setattr(collector, "measure", fake_measure)
    monkeypatch.setattr(collector, "read_device_state", lambda target: state)
    monkeypatch.setattr(collector.RPCExecutor, "connect", lambda *a: connects.append(a) or executor())
    monkeypatch.setattr(collector, "BATCH_SIZE", 2)
    monkeypatch.setattr(collector.time, "sleep", sleeps.append)
    args = argparse.Namespace(
        positions_from=source / "pool.parquet", order="test_first", splits=None,
        rpc_host="127.0.0.1", rpc_port=9090, ssh_target="nano", output_dir=out,
        n=5000, smoke=False, pilot_only=False, allow_dirty=True,
    )

    collector.collect_from_positions(args, argparse.ArgumentParser(), profile, workload)

    pool = pd.read_parquet(out / "pool.parquet")
    assert pool["id"].tolist() == [12, 10, 13, 11, 14]
    assert pool["batch_id"].tolist() == [0, 0, 1, 1, 2]
    assert dict(zip(pool["id"], pool["status"])) == {
        12: "ok", 10: "ok", 13: "ok", 11: "ok", 14: "rpc_error"
    }
    m2_columns = {"id", "sketch_id", "vector", "repeats_ms", "median_ms", "std_ms",
                  "status", "build_time_s", "batch_id", "timestamp"}
    assert m2_columns | {"target_id", "device_state", "upload_time_s", "link_load_time_s"} <= set(pool)
    assert set(pool["target_id"]) == {"jetson_nano"}
    assert json.loads(pool["device_state"].iloc[0])["before"]["rpc_servers"][0]["pid"] == 1
    sessions = [json.loads(s)["rpc_session_index"] for s in pool["device_state"]]
    assert sessions[0] == {"batch_start": 0, "batch_end": 0}
    assert sessions[3] == {"batch_start": 0, "batch_end": 1}  # id 11 retried once
    assert sessions[4] == {"batch_start": 1, "batch_end": 3}  # id 14 failed twice
    assert calls.count(11) == 2 and calls.count(14) == 2  # one retry each
    assert len(connects) == 1 + 1 + 2  # initial + retry for 11 + retry and recovery for 14
    assert sleeps == [collector.COOLDOWN_S, collector.COOLDOWN_S]
    canary_log = pd.read_parquet(out / "canary_log.parquet")
    assert canary_log["batch_id"].tolist() == [0, 1, 2]
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["rpc_server"]["environment"]["TVM_NUM_THREADS"] == "4"  # preserved
    assert meta["collection"]["status_counts"] == {"ok": 4, "rpc_error": 1}
    assert meta["collection"]["collection_complete"] is True
    assert meta["collection"]["reconnects"] == 3
    assert "git_commit" in meta["collection"]["provenance"]
    assert len((out / "collection_errors.jsonl").read_text().splitlines()) == 3

    calls.clear()
    collector.collect_from_positions(args, argparse.ArgumentParser(), profile, workload)
    assert calls == []  # resumable: nothing is re-measured
    assert len(pd.read_parquet(out / "pool.parquet")) == 5

    state["rpc_servers"][0]["environment"]["TVM_NUM_THREADS"] = "1"
    (out / "pool.parquet").unlink()
    with pytest.raises(RuntimeError, match="RPC server environment differs"):
        collector.collect_from_positions(args, argparse.ArgumentParser(), profile, workload)
