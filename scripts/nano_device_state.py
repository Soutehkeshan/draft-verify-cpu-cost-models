#!/usr/bin/env python3
"""Print one read-only JSON snapshot of Nano CPU clock, thermal, power and load state.

Runs on the Nano's system Python 3.6 (piped over SSH stdin); it only reads
sysfs/procfs and runs `nvpmodel -q` / `jetson_clocks --show`. It never changes
clocks, governors, power modes or fan settings. Taken only between timing runs.
"""
import glob
import json
import os
import re
import subprocess
import time


def _read(path):
    try:
        with open(path) as source:
            return source.read().strip()
    except (IOError, OSError):
        return None


def _int(path):
    value = _read(path)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run(command):
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=10,
        )
        return completed.returncode, completed.stdout
    except (OSError, subprocess.SubprocessError) as error:
        return None, str(error)


def _cpus():
    cpus = {}
    for directory in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*")):
        name = os.path.basename(directory)
        freq = directory + "/cpufreq/"
        online = _read(directory + "/online")
        cpus[name] = {
            "online": online if online is not None else "1",
            "governor": _read(freq + "scaling_governor"),
            "cur_khz": _int(freq + "scaling_cur_freq"),
            "min_khz": _int(freq + "scaling_min_freq"),
            "max_khz": _int(freq + "scaling_max_freq"),
            "hw_min_khz": _int(freq + "cpuinfo_min_freq"),
            "hw_max_khz": _int(freq + "cpuinfo_max_freq"),
        }
    return cpus


def _time_in_state():
    """Cumulative residency per frequency (kernel units: 10 ms) for each policy."""
    policies = {}
    for directory in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*")):
        table = _read(directory + "/stats/time_in_state")
        related = _read(directory + "/related_cpus")
        if table is None:
            continue
        policies[os.path.basename(directory)] = {
            "related_cpus": related,
            "ticks_10ms": {
                line.split()[0]: int(line.split()[1])
                for line in table.splitlines()
                if len(line.split()) == 2
            },
            "total_transitions": _int(directory + "/stats/total_trans"),
        }
    return policies


def _thermal():
    zones = {}
    for directory in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        zone_type = _read(directory + "/type") or os.path.basename(directory)
        zones[zone_type] = _int(directory + "/temp")
    return zones


def _cooling():
    devices = {}
    for directory in sorted(glob.glob("/sys/class/thermal/cooling_device*")):
        device_type = _read(directory + "/type") or os.path.basename(directory)
        key = device_type if device_type not in devices else os.path.basename(directory)
        devices[key] = {
            "cur_state": _int(directory + "/cur_state"),
            "max_state": _int(directory + "/max_state"),
        }
    return devices


def _nvpmodel():
    code, output = _run(["nvpmodel", "-q"])
    mode = re.search(r"NV Power Mode:\s*(\S+)", output or "")
    mode_id = [line for line in (output or "").splitlines() if line.strip().isdigit()]
    return {
        "returncode": code,
        "mode": mode.group(1) if mode else None,
        "mode_id": int(mode_id[0]) if mode_id else None,
    }


def _jetson_clocks(cpus):
    code, output = _run(["jetson_clocks", "--show"])
    needs_root = "root" in (output or "").lower()
    pinned = [
        info["min_khz"] is not None
        and info["min_khz"] == info["max_khz"] == info["hw_max_khz"]
        for info in cpus.values()
        if info["online"] == "1"
    ]
    return {
        "show_returncode": code,
        "show_available": not needs_root and code == 0,
        "show_output": None if needs_root else (output or "")[:2000],
        "show_unavailable_reason": "requires root" if needs_root else None,
        "inferred_cpu_clocks_pinned": bool(pinned) and all(pinned),
        "inference": "jetson_clocks pins scaling_min_freq = scaling_max_freq = cpuinfo_max_freq",
    }


def _rpc_servers():
    servers = []
    for directory in glob.glob("/proc/[0-9]*"):
        cmdline = _read(directory + "/cmdline")
        if not cmdline or "tvm.exec.rpc_server" not in cmdline:
            continue
        environ = _read(directory + "/environ") or ""
        environment = dict(
            item.split("=", 1)
            for item in environ.split("\0")
            if "=" in item and (item.startswith("TVM_") or item.startswith("OMP_"))
        )
        servers.append(
            {
                "pid": int(os.path.basename(directory)),
                "args": cmdline.replace("\0", " ").strip(),
                "environment": environment,
            }
        )
    return sorted(servers, key=lambda item: item["pid"])


def _loadavg():
    fields = (_read("/proc/loadavg") or "").split()
    if len(fields) < 4:
        return None
    running, total = fields[3].split("/")
    return {
        "load1": float(fields[0]),
        "load5": float(fields[1]),
        "load15": float(fields[2]),
        "running": int(running),
        "total": int(total),
    }


def _top_processes():
    code, output = _run(["ps", "-eo", "pcpu,pid,comm", "--sort=-pcpu"])
    if code != 0:
        return None
    return [line.strip() for line in output.splitlines()[1:6]]


def main():
    cpus = _cpus()
    meminfo = _read("/proc/meminfo") or ""
    available = re.search(r"MemAvailable:\s+(\d+)", meminfo)
    snapshot = {
        "time_unix": time.time(),
        "uptime_s": float((_read("/proc/uptime") or "0").split()[0]),
        "cpus": cpus,
        "cpufreq_time_in_state": _time_in_state(),
        "thermal_millic": _thermal(),
        "cooling_devices": _cooling(),
        "nvpmodel": _nvpmodel(),
        "jetson_clocks": _jetson_clocks(cpus),
        "loadavg": _loadavg(),
        "mem_available_kb": int(available.group(1)) if available else None,
        "top_processes": _top_processes(),
        "rpc_servers": _rpc_servers(),
        "emc_frequency": "not readable without root (debugfs)",
    }
    print(json.dumps(snapshot, sort_keys=True))


if __name__ == "__main__":
    main()
