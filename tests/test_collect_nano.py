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
M2_COLUMNS = {"id", "sketch_id", "vector", "repeats_ms", "median_ms", "std_ms",
              "status", "build_time_s", "batch_id", "timestamp"}


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


class FakeProbe:
    """Read-only probe stand-in: each call advances 1000 s of residency at `khz`."""

    def __init__(self):
        self.khz = "1479000"
        self.ticks = {"1428000": 0, "1479000": 0}
        self.environment = dict(SERVER_ENV)

    def __call__(self, ssh_target):
        self.ticks[self.khz] += 100_000
        return {
            "cpus": {"cpu0": {"online": "1", "cur_khz": 1479000, "governor": "schedutil",
                              "min_khz": 102000, "max_khz": 1479000, "hw_max_khz": 1479000}},
            "cpufreq_time_in_state": {"policy0": {"related_cpus": "0 1 2 3",
                                                  "ticks_10ms": dict(self.ticks),
                                                  "total_transitions": 0}},
            "thermal_millic": {"CPU-therm": 31000, "PMIC-Die": 100000},
            "cooling_devices": {},
            "rpc_servers": [{"pid": 1, "args": "rpc_server", "environment": dict(self.environment)}],
        }


@pytest.fixture
def positions_run(tmp_path, monkeypatch):
    """Five valid positions, real layouts and order logic; fake RPC, probe and timing."""
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
        {"test": [12, 10], "validation": [13], "train": [11, 14],
         "nested_train": {"1": [14], "all": [14, 11]}},
    )
    _qualified_output(
        collector, out,
        canaries=[{"sketch_id": 12 % 3, "vector": vectors[12], "id": 12, "pilot_median_ms": 1.1}],
    )
    by_vector = {tuple(v): i for i, v in vectors.items()}
    run = argparse.Namespace(
        collector=collector, profile=profile, workload=workload, out=out,
        failures={}, statuses={}, canary_ms=1.1, calls=[], connects=[], sleeps=[],
        probe=FakeProbe(),
    )

    def fake_measure(position, sketch, layout, config, executor):
        assert config.slow_threshold_ms == 700.0
        sample_id = by_vector[tuple(int(v) for v in position)]  # canary reuses id 12's vector
        run.calls.append(sample_id)
        if run.failures.get(sample_id, 0):
            run.failures[sample_id] -= 1
            raise ConnectionResetError("RPC session dropped")
        status = run.statuses.get(sample_id, "ok")
        median = run.canary_ms if sample_id == 12 else 1.0
        return MeasurementResult(TimingResult((median,) * 5, median, 0.01, status, 0.0),
                                 0.1, 0.2, 0.3, 0.4)

    executor = type("Executor", (), {"input_prepare_time_s": 0.0, "input_upload_time_s": 0.0})
    monkeypatch.setattr(collector, "measure", fake_measure)
    monkeypatch.setattr(collector, "read_device_state", run.probe)
    monkeypatch.setattr(
        collector.RPCExecutor, "connect", lambda *a: run.connects.append(a) or executor()
    )
    monkeypatch.setattr(collector.time, "sleep", run.sleeps.append)
    run.args = argparse.Namespace(
        positions_from=source / "pool.parquet", order="test_first", splits=None,
        rpc_host="127.0.0.1", rpc_port=9090, ssh_target="nano", output_dir=out,
        n=5000, smoke=False, pilot_only=False, allow_dirty=True, batch_size=2, cooldown_s=60.0,
    )
    run.collect = lambda: collector.collect_from_positions(
        run.args, argparse.ArgumentParser(), profile, workload
    )
    return run


def test_test_first_order_is_test_then_validation_then_nested_train():
    collector = _collector()
    splits = {
        "test": [9, 3],
        "validation": [7, 1],
        "train": [2, 4, 5, 6, 8],
        "nested_train": {"2": [5, 2], "4": [5, 2, 8, 4], "all": [5, 2, 8, 4, 6]},
    }
    assert collector.order_test_first(splits, set(range(1, 10))) == [9, 3, 7, 1, 5, 2, 8, 4, 6]
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
    n_test, n_val = len(splits["test"]), len(splits["validation"])
    assert len(order) == len(ids) == len(set(order))
    assert order[:n_test] == splits["test"]
    assert order[n_test : n_test + n_val] == splits["validation"]
    assert order[n_test + n_val : n_test + n_val + 100] == splits["nested_train"]["100"]
    assert order[n_test + n_val :] == splits["nested_train"]["all"]


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


def test_sampling_path_is_reserved_for_the_m2_target(monkeypatch, capsys):
    collector = _collector()
    monkeypatch.setattr(sys, "argv", ["01_collect.py", "--target", "jetson_nano"])
    monkeypatch.setattr(collector.RPCExecutor, "connect", lambda *_: pytest.fail("must not connect"))
    with pytest.raises(SystemExit) as stopped:
        collector.main()
    assert stopped.value.code == 2
    assert "--positions-from" in capsys.readouterr().err


def test_positions_collection_checkpoints_per_batch_resumes_and_retries_once(positions_run):
    run = positions_run
    run.failures[11] = 1  # one dropped RPC call, recovered by the single retry

    run.collect()

    pool = pd.read_parquet(run.out / "pool.parquet")
    assert pool["id"].tolist() == [12, 10, 13, 14, 11]  # TEST, VALIDATION, nested TRAIN
    assert pool["batch_id"].tolist() == [0, 0, 1, 1, 2]
    assert set(pool["status"]) == {"ok"}
    assert M2_COLUMNS | {"target_id", "device_state", "upload_time_s", "link_load_time_s"} <= set(pool)
    assert set(pool["target_id"]) == {"jetson_nano"}
    state = json.loads(pool["device_state"].iloc[0])
    assert state["before"]["rpc_servers"][0]["pid"] == 1
    assert state["frequency_residency"]["policy0"]["seconds_at_max"] == 1000.0
    sessions = [json.loads(s)["rpc_session_index"] for s in pool["device_state"]]
    assert sessions[:4] == [{"batch_start": 0, "batch_end": 0}] * 4
    assert sessions[4] == {"batch_start": 0, "batch_end": 1}  # id 11 retried once
    assert run.calls.count(11) == 2
    assert len(run.connects) == 2
    assert run.sleeps == [60.0, 60.0]
    canary_log = pd.read_parquet(run.out / "canary_log.parquet")
    assert canary_log["batch_id"].tolist() == [0, 1, 2]
    batches = [json.loads(line) for line in (run.out / "collection_batches.jsonl").read_text().splitlines()]
    assert [b["canary_drift_pct"] for b in batches] == [0.0, 0.0, 0.0]
    assert all(not b["frequency"]["drop"] and b["pause_reason"] is None for b in batches)
    assert batches[0]["temperature_c"] == {"min": 31.0, "max": 31.0}  # PMIC-Die placeholder ignored
    meta = json.loads((run.out / "run_meta.json").read_text())
    assert meta["rpc_server"]["environment"]["TVM_NUM_THREADS"] == "4"  # preserved
    collection = meta["collection"]
    assert collection["state"] == "complete" and collection["collection_complete"] is True
    assert collection["status_counts"] == {"ok": 5}
    assert collection["reconnects"] == 1
    assert collection["batch_size"] == 2 and collection["cooldown_s"] == 60.0
    assert collection["git_commit"] == collection["provenance"]["git_commit"]
    assert collection["canary_baseline"] == "pilot median (canary.json)"

    run.calls.clear()
    run.collect()
    assert run.calls == []  # resumable: nothing is re-measured
    assert len(pd.read_parquet(run.out / "pool.parquet")) == 5


def _paused(run) -> dict:
    with pytest.raises(SystemExit) as stopped:
        run.collect()
    assert stopped.value.code == run.collector.PAUSE_EXIT_CODE
    return json.loads((run.out / "run_meta.json").read_text())["collection"]


def test_two_reconnects_in_a_row_pause_after_checkpointing_rpc_error(positions_run):
    run = positions_run
    run.failures[13] = 2  # first VALIDATION row fails twice
    collection = _paused(run)
    assert collection["state"] == "paused"
    assert "2 RPC reconnects in a row" in collection["pause_reason"]
    pool = pd.read_parquet(run.out / "pool.parquet")
    assert dict(zip(pool["id"], pool["status"])) == {12: "ok", 10: "ok", 13: "rpc_error"}


def test_canary_drift_in_two_consecutive_batches_pauses_before_measuring(positions_run):
    run = positions_run
    run.canary_ms = 1.2  # +9.1% vs the 1.1 ms baseline
    collection = _paused(run)
    assert "canary median drift +9.09%" in collection["pause_reason"]
    pool = pd.read_parquet(run.out / "pool.parquet")
    assert pool["batch_id"].tolist() == [0, 0]  # second drifted batch measured canaries only
    assert len(pd.read_parquet(run.out / "canary_log.parquet")) == 2


def test_frequency_drop_under_load_pauses(positions_run):
    run = positions_run
    run.probe.khz = "1428000"  # no residency at the 1479 MHz maximum
    collection = _paused(run)
    assert collection["pause_reason"].startswith("frequency drop under load")
    assert len(pd.read_parquet(run.out / "pool.parquet")) == 2


def test_non_ok_rows_above_five_percent_pause(positions_run):
    run = positions_run
    run.statuses[10] = "slow_skip"
    collection = _paused(run)
    assert "1/2 non-ok rows" in collection["pause_reason"]


def test_server_environment_change_pauses_before_the_batch(positions_run):
    run = positions_run
    run.probe.environment["TVM_NUM_THREADS"] = "1"
    collection = _paused(run)
    assert "RPC server environment differs" in collection["pause_reason"]
    assert not (run.out / "pool.parquet").exists()
