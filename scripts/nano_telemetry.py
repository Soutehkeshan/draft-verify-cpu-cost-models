#!/usr/bin/env python3
"""Sample Jetson Nano CPU telemetry and accept timing-window markers on stdin."""
from __future__ import print_function

import argparse
import glob
import json
import os
import select
import subprocess
import sys
import time


def _emit(payload):
    print(json.dumps(payload, sort_keys=True), flush=True)


def _read_process_table(proc_root="/proc"):
    processes = []
    for name in os.listdir(proc_root):
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, name, "cmdline"), "rb") as source:
                argv = [
                    item.decode("utf-8", "replace")
                    for item in source.read().split(b"\0")
                    if item
                ]
            with open(os.path.join(proc_root, name, "status"), "r") as source:
                status = source.readlines()
        except IOError:
            continue
        ppid = None
        for line in status:
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
                break
        if ppid is not None:
            processes.append({"pid": int(name), "ppid": ppid, "argv": argv})
    return processes


def _option_value(argv, option, default):
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
    return default


def _select_rpc_processes(processes, rpc_port, session_pid=None):
    servers = [
        process
        for process in processes
        if "tvm.exec.rpc_server" in process["argv"]
        and _option_value(process["argv"], "--port", "9090") == str(rpc_port)
    ]
    if len(servers) != 1:
        raise RuntimeError(
            "Expected one TVM RPC server on port {}; found {}".format(
                rpc_port, len(servers)
            )
        )
    server = servers[0]
    listeners = [
        process
        for process in processes
        if process["ppid"] == server["pid"]
        and "tvm.exec.popen_worker" in process["argv"]
    ]
    if len(listeners) != 1:
        raise RuntimeError(
            "Expected one TVM RPC listener worker under PID {}; found {}".format(
                server["pid"], len(listeners)
            )
        )
    listener = listeners[0]
    sessions = [
        process
        for process in processes
        if process["ppid"] == listener["pid"]
    ]
    if session_pid is not None:
        sessions = [
            process for process in sessions if process["pid"] == session_pid
        ]
        if not sessions:
            raise RuntimeError(
                "PID {} is not an active RPC session child of PID {}".format(
                    session_pid, listener["pid"]
                )
            )
    if len(sessions) != 1:
        raise RuntimeError(
            "Expected one active RPC session process under PID {}; found {}. "
            "Use --pid when more than one RPC client is connected.".format(
                listener["pid"], len(sessions)
            )
        )
    return server, listener, sessions[0]


def _rpc_environment(pid):
    wanted = {
        b"TVM_NUM_THREADS",
        b"OMP_NUM_THREADS",
        b"TVM_BIND_THREADS",
        b"TVM_EXCLUDE_WORKER0",
        b"TVM_THREAD_POOL_SPIN_COUNT",
    }
    values = {}
    try:
        with open("/proc/{}/environ".format(pid), "rb") as source:
            entries = source.read().split(b"\0")
    except IOError:
        return values
    for entry in entries:
        key, separator, value = entry.partition(b"=")
        if separator and key in wanted:
            values[key.decode("ascii")] = value.decode("ascii", "replace")
    return values


def _parse_rpc_thread_counters(stat, schedstat):
    right_paren = stat.rfind(")")
    if right_paren < 0:
        return None
    fields = stat[right_paren + 2 :].split()
    sched_fields = schedstat.split()
    if len(fields) <= 36 or not sched_fields:
        return None
    try:
        runtime_ns = int(sched_fields[0])
        processor = int(fields[36])
    except ValueError:
        return None
    return runtime_ns, processor


def _read_rpc_threads(pid):
    threads = {}
    task_dir = "/proc/{}/task".format(pid)
    try:
        tids = os.listdir(task_dir)
    except IOError:
        return threads
    for tid_text in tids:
        try:
            with open(os.path.join(task_dir, tid_text, "stat"), "r") as source:
                stat = source.read()
            with open(os.path.join(task_dir, tid_text, "schedstat"), "r") as source:
                schedstat = source.read()
        except IOError:
            continue
        counters = _parse_rpc_thread_counters(stat, schedstat)
        if counters is not None:
            threads[int(tid_text)] = counters
    return threads


def _read_cpu_counters():
    counters = {}
    try:
        with open("/proc/stat", "r") as source:
            lines = source.readlines()
    except IOError:
        return counters
    for line in lines:
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
            continue
        try:
            core = int(fields[0][3:])
            values = [int(value) for value in fields[1:]]
        except ValueError:
            continue
        if len(values) >= 4:
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            counters[core] = (sum(values[:8]), idle)
    return counters


def _cpu_utilization(previous, current):
    result = {}
    for core, (total, idle) in current.items():
        if core not in previous:
            result[core] = None
            continue
        previous_total, previous_idle = previous[core]
        total_delta = total - previous_total
        idle_delta = idle - previous_idle
        result[core] = (
            100.0 * (total_delta - idle_delta) / total_delta
            if total_delta > 0
            else None
        )
    return result


def _read_frequency_and_governor(core):
    root = "/sys/devices/system/cpu/cpu{}/cpufreq".format(core)
    frequency = None
    frequency_source = None
    for filename in ("scaling_cur_freq", "cpuinfo_cur_freq"):
        try:
            with open(os.path.join(root, filename), "r") as source:
                frequency = int(source.read().strip())
            frequency_source = filename
            break
        except (IOError, ValueError):
            continue
    try:
        with open(os.path.join(root, "scaling_governor"), "r") as source:
            governor = source.read().strip()
    except IOError:
        governor = None
    return frequency, governor, frequency_source


def _read_hardware_status(current_cpu):
    frequencies = {}
    governors = {}
    frequency_sources = {}
    for core in sorted(current_cpu):
        frequency, governor, frequency_source = _read_frequency_and_governor(core)
        frequencies[core] = frequency
        governors[core] = governor
        frequency_sources[core] = frequency_source
    return frequencies, governors, frequency_sources, _read_temperatures()


def _read_temperatures():
    temperatures = {}
    for path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        zone = path.split("/")[-2]
        try:
            with open(path, "r") as source:
                raw = int(source.read().strip())
            with open(os.path.join(os.path.dirname(path), "type"), "r") as source:
                label = source.read().strip()
        except (IOError, ValueError):
            continue
        key = "{}:{}".format(zone, label)
        temperatures[key] = raw / 1000.0 if abs(raw) > 1000 else float(raw)
    return temperatures


def _power_mode():
    executable = None
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(directory, "nvpmodel")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            executable = candidate
            break
    if executable is None:
        return {"available": False, "output": None, "returncode": None}
    try:
        result = subprocess.run(
            [executable, "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "available": True,
            "output": str(error),
            "returncode": None,
        }
    return {
        "available": True,
        "output": (result.stdout + result.stderr).strip(),
        "returncode": result.returncode,
    }


def _sample(
    pid,
    previous_cpu,
    previous_threads,
    clock_ticks,
    active_measurement,
    sample_hardware,
    thread_only,
):
    started = time.monotonic()
    cpu_started = time.process_time()
    current_cpu = {} if thread_only else _read_cpu_counters()
    current_threads = _read_rpc_threads(pid)
    utilization = (
        {} if thread_only else _cpu_utilization(previous_cpu, current_cpu)
    )
    active_tids = []
    active_cpus = set()
    active_thread_runtime_delta_ns = {}
    for tid, (runtime_ns, processor) in current_threads.items():
        previous = previous_threads.get(tid)
        if previous is None:
            # Thread created after the previous baseline (e.g. a lazily-spawned TVM
            # worker pool thread).  Count it as active if it has accumulated any
            # runtime; use that full runtime as the within-window delta.
            if runtime_ns > 0:
                active_tids.append(tid)
                active_cpus.add(processor)
                active_thread_runtime_delta_ns[str(tid)] = runtime_ns
        elif runtime_ns > previous[0]:
            active_tids.append(tid)
            active_cpus.add(processor)
            active_thread_runtime_delta_ns[str(tid)] = runtime_ns - previous[0]
    if sample_hardware and not thread_only:
        frequencies, governors, frequency_sources, temperatures = (
            _read_hardware_status(current_cpu)
        )
    else:
        frequencies, governors, frequency_sources, temperatures = {}, {}, {}, {}
    completed = time.monotonic()
    return (
        {
            "type": "sample",
            "monotonic_ns": int(started * 1_000_000_000),
            "sample_duration_ms": (completed - started) * 1000.0,
            "sample_cpu_duration_ms": (
                time.process_time() - cpu_started
            ) * 1000.0,
            "active_measurement_id": active_measurement,
            "cpu_utilization_pct": utilization,
            "frequency_khz": frequencies,
            "frequency_source": frequency_sources,
            "governor": governors,
            "temperatures_c": temperatures,
            "rpc_server_thread_count": len(current_threads),
            "active_rpc_thread_count": len(active_tids),
            "active_rpc_non_session_thread_count": sum(
                1 for tid in active_tids if tid != pid
            ),
            "active_rpc_cpu_ids": sorted(active_cpus),
            "active_rpc_tids": sorted(active_tids),
            "active_rpc_thread_runtime_delta_ns": active_thread_runtime_delta_ns,
            "rpc_thread_runtime_ns": {
                str(tid): runtime_ns
                for tid, (runtime_ns, _) in current_threads.items()
            },
            "rpc_thread_processor_ids": {
                str(tid): processor
                for tid, (_, processor) in current_threads.items()
            },
            "hardware_sampled": sample_hardware and not thread_only,
            "thread_only": thread_only,
            "clock_ticks_per_second": clock_ticks,
        },
        current_cpu,
        current_threads,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--hardware-interval-ms", type=int, default=100)
    parser.add_argument("--rpc-port", type=int, default=9090)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--self-delete", action="store_true")
    parser.add_argument("--thread-only", action="store_true")
    args = parser.parse_args()
    if args.interval_ms < 5:
        parser.error("--interval-ms must be at least 5")
    if args.hardware_interval_ms < 50:
        parser.error("--hardware-interval-ms must be at least 50")
    if not 1 <= args.rpc_port <= 65535:
        parser.error("--rpc-port must be between 1 and 65535")

    server, listener, session = _select_rpc_processes(
        _read_process_table(), args.rpc_port, args.pid
    )
    pid = session["pid"]
    if not os.path.isdir("/proc/{}/task".format(pid)):
        raise RuntimeError("RPC session PID {} does not exist".format(pid))
    clock_ticks = int(os.sysconf("SC_CLK_TCK"))
    power_start = _power_mode()
    initial_threads = _read_rpc_threads(pid)
    _emit(
        {
            "type": "metadata",
            "rpc_server_pid": pid,
            "rpc_master_pid": server["pid"],
            "rpc_listener_pid": listener["pid"],
            "rpc_session_pid": pid,
            "rpc_server_thread_ids_at_start": sorted(initial_threads),
            "rpc_server_thread_count_at_start": len(initial_threads),
            "cpu_count": os.cpu_count(),
            "clock_ticks_per_second": clock_ticks,
            "system_cpu_counter_resolution_ms": 1000.0 / clock_ticks,
            "sample_interval_ms": args.interval_ms,
            "hardware_sample_interval_ms": args.hardware_interval_ms,
            "rpc_thread_activity_source": "proc_task_schedstat runtime_ns",
            "thread_only": args.thread_only,
            "rpc_server_thread_environment": _rpc_environment(pid),
            "power_mode_start": power_start,
        }
    )
    if args.self_delete:
        try:
            os.unlink(__file__)
        except OSError as error:
            _emit({"type": "helper_cleanup_warning", "message": str(error)})

    period = args.interval_ms / 1000.0
    hardware_period = args.hardware_interval_ms / 1000.0
    next_sample = time.monotonic()
    previous_cpu = {}
    previous_threads = {}
    previous_sample_time = None
    previous_hardware_sample_time = None
    sampler_started_wall = time.monotonic()
    sampler_started_cpu = time.process_time()
    active_measurement = None
    while True:
        timeout = max(0.0, next_sample - time.monotonic())
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            line = sys.stdin.readline()
            if not line:
                break
            command = json.loads(line)
            action = command.get("action")
            measurement_id = command.get("measurement_id")
            if action == "start":
                if active_measurement is not None:
                    raise RuntimeError("A telemetry timing window is already active")
                active_measurement = measurement_id
                previous_cpu = (
                    {} if args.thread_only else _read_cpu_counters()
                )
                previous_threads = _read_rpc_threads(pid)
                previous_sample_time = time.monotonic()
                previous_hardware_sample_time = None
                next_sample = time.monotonic() + period
            elif action == "end":
                if active_measurement != measurement_id:
                    raise RuntimeError(
                        "Timing marker mismatch: {!r} != {!r}".format(
                            active_measurement, measurement_id
                        )
                    )
                sample, previous_cpu, previous_threads = _sample(
                    pid,
                    previous_cpu,
                    previous_threads,
                    clock_ticks,
                    active_measurement,
                    previous_hardware_sample_time is None
                    or time.monotonic() - previous_hardware_sample_time
                    >= hardware_period,
                    args.thread_only,
                )
                sample_time = sample["monotonic_ns"] / 1_000_000_000.0
                if sample["hardware_sampled"]:
                    previous_hardware_sample_time = sample_time
                sample["interval_since_last_sample_ms"] = (
                    (sample_time - previous_sample_time) * 1000.0
                    if previous_sample_time is not None
                    else None
                )
                previous_sample_time = sample_time
                _emit(sample)
            elif action != "stop":
                raise RuntimeError("Unknown telemetry marker {!r}".format(action))
            _emit(
                {
                    "type": "marker",
                    "action": action,
                    "measurement_id": measurement_id,
                    "token": command.get("token"),
                    "monotonic_ns": int(time.monotonic() * 1_000_000_000),
                }
            )
            if action == "end":
                active_measurement = None
            if action == "stop":
                break

        now = time.monotonic()
        if now >= next_sample:
            sample_hardware = (
                active_measurement is not None
                and (
                    previous_hardware_sample_time is None
                    or now - previous_hardware_sample_time >= hardware_period
                )
            )
            sample, previous_cpu, previous_threads = _sample(
                pid,
                previous_cpu,
                previous_threads,
                clock_ticks,
                active_measurement,
                sample_hardware,
                args.thread_only,
            )
            sample_time = sample["monotonic_ns"] / 1_000_000_000.0
            if sample["hardware_sampled"]:
                previous_hardware_sample_time = sample_time
            sample["interval_since_last_sample_ms"] = (
                (sample_time - previous_sample_time) * 1000.0
                if previous_sample_time is not None
                else None
            )
            previous_sample_time = sample_time
            _emit(sample)
            next_sample = max(next_sample + period, now + period)

    _emit(
        {
            "type": "shutdown",
            "power_mode_end": _power_mode(),
            "sampler_elapsed_seconds": time.monotonic() - sampler_started_wall,
            "sampler_cpu_seconds": time.process_time() - sampler_started_cpu,
        }
    )


if __name__ == "__main__":
    main()
