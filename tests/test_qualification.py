"""Nano qualification helpers: device-state math, layout helpers, replay selection."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from dvcm.device_state import frequency_residency, summarize_device_state
from dvcm.encoding import layout_to_json, space_size, structural_layout, tile_choice_count

ROOT = Path(__file__).resolve().parents[1]


def _state(ticks: dict[str, int], transitions: int) -> dict:
    return {
        "time_unix": 1.0,
        "cpus": {
            "cpu0": {"online": "1", "cur_khz": 1479000, "governor": "schedutil",
                     "min_khz": 102000, "max_khz": 1479000},
            "cpu1": {"online": "0", "cur_khz": None, "governor": None,
                     "min_khz": None, "max_khz": None},
        },
        "cpufreq_time_in_state": {
            "policy0": {"related_cpus": "0 1 2 3", "ticks_10ms": ticks,
                        "total_transitions": transitions}
        },
        "cooling_devices": {"cpu-balanced": {"cur_state": 0, "max_state": 66},
                            "CORE-floor": {"cur_state": 1, "max_state": 1}},
        "thermal_millic": {"CPU-therm": 30000},
        "nvpmodel": {"mode": "MAXN"},
        "jetson_clocks": {"show_available": False, "inferred_cpu_clocks_pinned": False},
        "loadavg": {"load1": 0.1},
        "rpc_servers": [{"pid": 7, "args": "rpc_server", "environment": {"TVM_NUM_THREADS": "4"}}],
    }


def test_frequency_residency_is_the_time_in_state_delta():
    before = _state({"102000": 100, "1479000": 200}, 10)
    after = _state({"102000": 150, "1479000": 2200}, 16)
    policy = frequency_residency(before, after)["policy0"]
    assert policy["seconds_by_khz"] == {"102000": 0.5, "1479000": 20.0}
    assert policy["seconds_at_max"] == 20.0
    assert policy["total_s"] == 20.5
    assert policy["max_khz"] == 1479000
    assert abs(policy["fraction_at_max"] - 20.0 / 20.5) < 1e-12
    assert policy["transitions"] == 6


def test_device_state_summary_keeps_online_cpus_and_active_cooling_only():
    summary = summarize_device_state(_state({"1479000": 1}, 0))
    assert summary["cur_khz"] == {"cpu0": 1479000}
    assert summary["governors"] == ["schedutil"]
    assert summary["active_cooling_states"] == {"CORE-floor": 1}
    assert summary["rpc_servers"] == [{"pid": 7, "environment": {"TVM_NUM_THREADS": "4"}}]


def test_layout_helpers_count_the_512_matmul_space():
    tiles = [
        {"n": 4, "extent": 512, "total_exp": 9, "max_inner_exp": 6, "max_inner_factor": 64,
         "slice": slice(0, 4)},
        {"n": 4, "extent": 512, "total_exp": 9, "max_inner_exp": 6, "max_inner_factor": 64,
         "slice": slice(4, 8)},
        {"n": 2, "extent": 512, "total_exp": 9, "max_inner_exp": 6, "max_inner_factor": 64,
         "slice": slice(8, 10)},
    ]
    layout = {
        "tiles": tiles,
        "categoricals": [{"index": 10, "candidates": (0, 16, 64, 512)}],
        "dim": 11,
        "parallel_tile_levels": {"i": 2, "j": 2},
    }
    serialized = layout_to_json(layout, 0)
    assert serialized["tiles"][0]["slice"] == [0, 4]
    assert [tile_choice_count(t) for t in serialized["tiles"]] == [210, 210, 7]
    assert space_size(serialized) == space_size(layout) == 210 * 210 * 7 * 4 == 1_234_800
    with_target_values = {**serialized, "num_cores": 4, "vector_lanes": 4}
    assert structural_layout(with_target_values) == serialized


def test_replay_set_is_stratified_and_deterministic():
    spec = importlib.util.spec_from_file_location("qualify_nano_test", ROOT / "scripts" / "qualify_nano.py")
    qualify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualify)
    rng = np.random.default_rng(5)
    pool = pd.DataFrame(
        {
            "id": np.arange(600),
            "sketch_id": np.repeat([0, 1, 2], 200),
            "vector": [[0] * 11] * 600,
            "median_ms": rng.lognormal(1.0, 0.8, 600),
            "status": ["ok"] * 600,
        }
    )
    replay = qualify.select_replay_set(pool)
    assert len(replay) == 150 and replay["id"].is_unique
    assert replay.groupby(["sketch_id", "m2_decile"]).size().eq(5).all()
    assert sorted(replay["run_order"]) == list(range(150))
    assert replay.equals(qualify.select_replay_set(pool))
