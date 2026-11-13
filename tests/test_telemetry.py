from scripts.measure_nano_telemetry import _summarize_window
from scripts.nano_telemetry import (
    _cpu_utilization,
    _parse_rpc_thread_counters,
    _select_rpc_processes,
)


def test_cpu_utilization_uses_per_core_proc_stat_deltas():
    previous = {0: (100, 50), 1: (200, 180)}
    current = {0: (110, 55), 1: (220, 182)}

    assert _cpu_utilization(previous, current) == {0: 50.0, 1: 90.0}


def test_rpc_thread_activity_uses_high_resolution_schedstat_runtime():
    fields = ["R"] + ["0"] * 38
    fields[36] = "3"
    stat = "123 (python worker) " + " ".join(fields)

    assert _parse_rpc_thread_counters(stat, "12500000 20000 8") == (
        12_500_000,
        3,
    )


def test_rpc_process_discovery_selects_the_active_session_child():
    processes = [
        {
            "pid": 10,
            "ppid": 1,
            "argv": ["python3", "-m", "tvm.exec.rpc_server", "--port", "9091"],
        },
        {
            "pid": 20,
            "ppid": 10,
            "argv": ["python3", "-m", "tvm.exec.popen_worker", "5", "4"],
        },
        {
            "pid": 31,
            "ppid": 20,
            "argv": ["python3", "-m", "tvm.exec.popen_worker", "5", "4"],
        },
    ]

    server, listener, session = _select_rpc_processes(processes, 9091)

    assert (server["pid"], listener["pid"], session["pid"]) == (10, 20, 31)


def test_rpc_process_discovery_requires_a_pid_for_multiple_sessions():
    processes = [
        {
            "pid": 10,
            "ppid": 1,
            "argv": ["python3", "-m", "tvm.exec.rpc_server", "--port", "9091"],
        },
        {
            "pid": 20,
            "ppid": 10,
            "argv": ["python3", "-m", "tvm.exec.popen_worker", "5", "4"],
        },
        {
            "pid": 31,
            "ppid": 20,
            "argv": ["python3", "-m", "tvm.exec.popen_worker", "5", "4"],
        },
        {
            "pid": 32,
            "ppid": 20,
            "argv": ["python3", "-m", "tvm.exec.popen_worker", "5", "4"],
        },
    ]

    _, _, session = _select_rpc_processes(processes, 9091, session_pid=32)

    assert session["pid"] == 32


def test_telemetry_window_associates_samples_with_one_timing_batch():
    measurement_id = "sketch-1-batch-2"
    per_core = {"0": 60.0, "1": 70.0, "2": 80.0, "3": 90.0}
    frequencies = {str(core): 1_479_000 for core in range(4)}
    governors = {str(core): "schedutil" for core in range(4)}
    sample = {
        "type": "sample",
        "monotonic_ns": 150,
        "interval_since_last_sample_ms": 10.0,
        "active_measurement_id": measurement_id,
        "cpu_utilization_pct": per_core,
        "frequency_khz": frequencies,
        "frequency_source": {
            str(core): "scaling_cur_freq" for core in range(4)
        },
        "governor": governors,
        "temperatures_c": {"thermal_zone0:CPU-therm": 56.0},
        "active_rpc_non_session_thread_count": 4,
        "active_rpc_thread_count": 5,
        "active_rpc_cpu_ids": [0, 1, 2, 3],
    }
    report = {
        "markers": [
            {
                "action": "start",
                "measurement_id": measurement_id,
                "monotonic_ns": 100,
            },
            {
                "action": "end",
                "measurement_id": measurement_id,
                "monotonic_ns": 200,
            },
        ],
        "samples": [
            sample,
            {**sample, "active_measurement_id": "another-batch"},
            {**sample, "monotonic_ns": 250},
        ],
    }

    summary = _summarize_window(report, measurement_id, expected_cores=4)

    assert summary["sample_count"] == 1
    assert summary["mean_sample_interval_ms"] == 10.0
    assert summary["max_sample_interval_ms"] == 10.0
    assert summary["max_active_rpc_non_session_threads"] == 4
    assert summary["max_active_rpc_threads_including_server"] == 5
    assert summary["rpc_cpu_ids_observed"] == [0, 1, 2, 3]
    assert summary["all_four_rpc_cpu_ids_observed"] is True
    assert summary["max_temperature_c"] == {"thermal_zone0:CPU-therm": 56.0}
    assert summary["per_core"]["3"]["mean_frequency_mhz"] == 1479.0
    assert summary["per_core"]["3"]["frequency_sources"] == ["scaling_cur_freq"]
