"""scripts/retime_nano_test.py: one-session TEST re-timing; no device is used."""
import hashlib
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path

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
RECORDED_ENV = {**SERVER_ENV, "CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++"}
TEST_IDS = [12, 10, 14]


def _module():
    spec = importlib.util.spec_from_file_location("retime_test", ROOT / "scripts" / "retime_nano_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeProbe:
    """Device-state stand-in: each call advances 1000 s of residency at the top clock."""

    def __init__(self):
        self.ticks = {"1479000": 0}
        self.pid = 1
        self.environment = dict(SERVER_ENV)

    def __call__(self, ssh_target):
        self.ticks["1479000"] += 100_000
        return {
            "cpus": {"cpu0": {"online": "1", "cur_khz": 1479000, "governor": "schedutil",
                              "min_khz": 102000, "max_khz": 1479000, "hw_max_khz": 1479000}},
            "cpufreq_time_in_state": {"policy0": {"related_cpus": "0 1 2 3",
                                                  "ticks_10ms": dict(self.ticks),
                                                  "total_transitions": 0}},
            "thermal_millic": {"CPU-therm": 31000},
            "cooling_devices": {},
            "rpc_servers": [{"pid": self.pid, "args": "rpc_server", "environment": dict(self.environment)}],
        }


@pytest.fixture
def retime_run(tmp_path, monkeypatch):
    """Real layouts and positions; fake RPC session, probe, environment read and timing."""
    module = _module()
    profile = replace(load_target_config("jetson_nano"), schedule_rules="llvm", postprocessors="llvm")
    workload = load_workload_config()
    context = module.COLLECT.ms.TuneContext(
        mod=module.COLLECT.make_matmul(workload.n),
        target=profile.make_tvm_target(),
        space_generator=module.COLLECT.PostOrderApply(sch_rules="llvm", postprocs="llvm"),
        rand_state=workload.seed,
        num_threads=4,
    )
    layouts = [describe_space(sketch.trace) for sketch in context.generate_design_space()]
    data = tmp_path / "nano"
    data.mkdir()
    for sketch_id, layout in enumerate(layouts):
        (data / f"layout_{sketch_id}.json").write_text(json.dumps(layout_to_json(layout, sketch_id)))
    rng = np.random.default_rng(3)
    ids = [10, 11, 12, 13, 14]
    vectors = {i: random_position(rng, layouts[i % 3]).astype(int).tolist() for i in ids}
    pd.DataFrame(
        {"id": ids, "sketch_id": [i % 3 for i in ids], "vector": [vectors[i] for i in ids],
         "repeats_ms": [[1.0] * 5] * 5, "median_ms": [1.0] * 5, "status": ["ok"] * 5}
    ).to_parquet(data / "pool.parquet", index=False)
    (data / "splits.json").write_text(json.dumps(
        {"test": TEST_IDS, "validation": [13], "train": [11],
         "nested_train": {"all": [11]}}
    ))
    (data / "run_meta.json").write_text(json.dumps(
        {"slow_threshold_ms": 700.0,
         "rpc_server": {"environment": RECORDED_ENV, "server_pid": 1}}
    ))
    (data / "canary.json").write_text(json.dumps(
        [{"sketch_id": 12 % 3, "vector": vectors[12], "id": 12, "pilot_median_ms": 1.1}]
    ))
    by_vector = {tuple(v): i for i, v in vectors.items()}
    state = {"calls": [], "connects": [], "sleeps": [], "fail_on": None, "probe": FakeProbe(),
             "environment": dict(RECORDED_ENV), "configs": []}

    def fake_measure(position, sketch, layout, config, executor):
        state["configs"].append(config)
        sample_id = by_vector[tuple(int(v) for v in position)]  # the canary reuses id 12's vector
        state["calls"].append(sample_id)
        if state["fail_on"] == sample_id:
            raise ConnectionResetError("RPC session dropped")
        median = 1.1 if sample_id == 12 else 1.0  # canary at its 1.1 ms baseline: no drift
        return MeasurementResult(
            TimingResult((median,) * config.repeats, median, 0.01, "ok", 0.0), 0.1, 0.2, 0.3, 0.4
        )

    executor = type("Executor", (), {"input_prepare_time_s": 0.0, "input_upload_time_s": 0.0})
    monkeypatch.setattr(module, "measure", fake_measure)
    monkeypatch.setattr(module, "read_device_state", state["probe"])
    monkeypatch.setattr(module, "read_server_environment", lambda target, pid: dict(state["environment"]))
    monkeypatch.setattr(module.RPCExecutor, "connect", lambda *a: state["connects"].append(a) or executor())
    monkeypatch.setattr(module.time, "sleep", state["sleeps"].append)
    argv = ["--data-dir", str(data), "--rpc-host", "127.0.0.1", "--ssh-target", "nano",
            "--batch-size", "2", "--allow-dirty"]
    return module, data, state, argv


def test_every_test_schedule_is_retimed_once_with_15_repeats_in_one_session(retime_run):
    module, data, state, argv = retime_run
    pool_sha = _sha256(data / "pool.parquet")

    module.main(argv)

    retimed = pd.read_parquet(data / "test_retime.parquet")
    meta = json.loads((data / "test_retime_meta.json").read_text())
    assert sorted(retimed["id"]) == sorted(TEST_IDS)
    assert retimed["id"].tolist() == meta["measurement_order"]  # seeded order, recorded
    assert meta["measurement_order"] == module.measurement_order(TEST_IDS, meta["order_seed"])
    assert retimed["retime_order"].tolist() == list(range(len(TEST_IDS)))
    assert {len(r) for r in retimed["repeats_ms"]} == {15}
    assert set(retimed["status"]) == {"ok"}
    assert set(retimed["rpc_session_index"]) == {0}
    assert all(config.repeats == 15 and math.isinf(config.slow_threshold_ms) for config in state["configs"])
    assert len(state["connects"]) == 1
    assert state["sleeps"] == [60.0]  # two batches of at most two schedules
    assert len(pd.read_parquet(data / "test_retime_canary.parquet")) == 2
    assert meta["state"] == "complete"
    assert meta["rpc_sessions_opened"] == 1 and meta["reconnects"] == 0
    assert meta["repeats"] == 15 and meta["slow_skip"] == "disabled"
    assert meta["inputs_sha256_before"] == meta["inputs_sha256_after"]
    assert meta["provenance"]["git_commit"]
    assert _sha256(data / "pool.parquet") == pool_sha
    assert not (data / "test_retime.partial.parquet").exists()


def test_existing_output_is_never_overwritten(retime_run):
    module, data, state, argv = retime_run
    (data / "test_retime.parquet").write_bytes(b"keep")

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        module.main(argv)

    assert (data / "test_retime.parquet").read_bytes() == b"keep"
    assert state["connects"] == []


def test_rpc_failure_stops_the_run_without_a_second_session(retime_run):
    module, data, state, argv = retime_run
    order = module.measurement_order(TEST_IDS, module.ORDER_SEED)
    state["fail_on"] = order[-1]  # the last schedule of the second batch

    with pytest.raises(SystemExit) as stopped:
        module.main(argv)

    assert stopped.value.code == module.STOP_EXIT_CODE
    assert len(state["connects"]) == 1
    assert not (data / "test_retime.parquet").exists()
    meta = json.loads((data / "test_retime_meta.json").read_text())
    assert meta["state"] == "stopped" and "ConnectionResetError" in meta["stop_reason"]
    partial = pd.read_parquet(data / "test_retime.partial.parquet")
    assert partial["id"].tolist() == order[:-1]


def test_a_different_server_process_or_environment_refuses_before_connecting(retime_run):
    module, data, state, argv = retime_run
    state["probe"].pid = 2
    with pytest.raises(SystemExit, match="RPC server"):
        module.main(argv)
    assert state["connects"] == []

    state["probe"].pid = 1
    state["environment"]["CXX"] = "/usr/bin/clang++"
    with pytest.raises(SystemExit, match="environment"):
        module.main(argv)
    assert state["connects"] == []
