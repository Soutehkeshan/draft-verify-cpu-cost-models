"""Read-only Nano device-state snapshots, taken between (never during) timing runs."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = ROOT / "scripts" / "nano_device_state.py"
TICKS_PER_SECOND = 100  # cpufreq stats time_in_state is reported in 10 ms units


def read_device_state(ssh_target: str, timeout_s: float = 30.0) -> dict[str, Any]:
    """Run the read-only probe on the Nano's system Python via SSH stdin."""
    completed = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            ssh_target,
            "python3",
            "-",
        ],
        input=PROBE_SCRIPT.read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Device-state probe failed ({completed.returncode}): "
            f"{completed.stderr.strip()[-500:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError("Device-state probe returned no JSON snapshot")
    return json.loads(lines[-1])


def frequency_residency(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Seconds spent at each CPU frequency between two snapshots, per cpufreq policy."""
    result: dict[str, Any] = {}
    for policy, start in before.get("cpufreq_time_in_state", {}).items():
        end = after.get("cpufreq_time_in_state", {}).get(policy)
        if end is None:
            continue
        seconds = {
            int(freq): (end["ticks_10ms"].get(freq, 0) - ticks) / TICKS_PER_SECOND
            for freq, ticks in start["ticks_10ms"].items()
        }
        total = sum(seconds.values())
        top = max(seconds) if seconds else None
        result[policy] = {
            "related_cpus": start.get("related_cpus"),
            "seconds_by_khz": {str(freq): value for freq, value in sorted(seconds.items())},
            "total_s": total,
            "max_khz": top,
            "seconds_at_max": seconds.get(top, 0.0) if top is not None else 0.0,
            "fraction_at_max": (seconds.get(top, 0.0) / total) if total > 0 else None,
            "mean_khz": (
                sum(freq * value for freq, value in seconds.items()) / total
                if total > 0
                else None
            ),
            "transitions": (
                (end.get("total_transitions") or 0) - (start.get("total_transitions") or 0)
            ),
        }
    return result


def summarize_device_state(state: dict[str, Any]) -> dict[str, Any]:
    """Compact per-snapshot record for logs and per-batch pool metadata."""
    online = {
        name: info for name, info in state.get("cpus", {}).items() if info.get("online") == "1"
    }
    cooling = {
        name: info["cur_state"]
        for name, info in state.get("cooling_devices", {}).items()
        if info.get("cur_state")
    }
    return {
        "time_unix": state.get("time_unix"),
        "cur_khz": {name: info.get("cur_khz") for name, info in online.items()},
        "governors": sorted({str(info.get("governor")) for info in online.values()}),
        "min_khz": sorted({info.get("min_khz") for info in online.values()}),
        "max_khz": sorted({info.get("max_khz") for info in online.values()}),
        "thermal_millic": state.get("thermal_millic"),
        "active_cooling_states": cooling,
        "nvpmodel": state.get("nvpmodel", {}).get("mode"),
        "jetson_clocks_show_available": state.get("jetson_clocks", {}).get("show_available"),
        "jetson_clocks_inferred_pinned": state.get("jetson_clocks", {}).get(
            "inferred_cpu_clocks_pinned"
        ),
        "loadavg": state.get("loadavg"),
        "mem_available_kb": state.get("mem_available_kb"),
        "rpc_servers": [
            {"pid": item["pid"], "environment": item["environment"]}
            for item in state.get("rpc_servers", [])
        ],
    }
