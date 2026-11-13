"""Stream read-only Nano CPU telemetry over an existing SSH connection."""
from __future__ import annotations

import json
import queue
import shlex
import subprocess
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any


class NanoTelemetry:
    """Run the stdlib Nano sampler and synchronize timing-window markers."""

    def __init__(
        self,
        ssh_target: str,
        interval_ms: int = 100,
        hardware_interval_ms: int = 100,
        rpc_pid: int | None = None,
        rpc_port: int = 9090,
        thread_only: bool = False,
    ) -> None:
        if not ssh_target or ssh_target.startswith("-") or any(
            character.isspace() for character in ssh_target
        ):
            raise ValueError("ssh_target must be an SSH alias or user@host")
        if interval_ms < 5:
            raise ValueError("interval_ms must be at least 5")
        if hardware_interval_ms < 50:
            raise ValueError("hardware_interval_ms must be at least 50")
        if rpc_pid is not None and rpc_pid < 1:
            raise ValueError("rpc_pid must be positive")
        if not 1 <= rpc_port <= 65535:
            raise ValueError("rpc_port must be between 1 and 65535")

        self.ssh_target = ssh_target
        self.interval_ms = interval_ms
        self.hardware_interval_ms = hardware_interval_ms
        self.rpc_pid = rpc_pid
        self.rpc_port = rpc_port
        self.thread_only = thread_only
        self.remote_script = "/tmp/dvcm_nano_telemetry_{}.py".format(
            uuid.uuid4().hex
        )
        self._process: subprocess.Popen[str] | None = None
        self._events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._all_events: list[dict[str, Any]] = []
        self._stderr_lines: list[str] = []
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._token = 0
        self.metadata: dict[str, Any] | None = None
        self.cleanup_error: str | None = None

    def start(self) -> NanoTelemetry:
        script = Path(__file__).resolve().parents[2] / "scripts" / "nano_telemetry.py"
        self._run_ssh_tool(
            [
                "scp",
                "-q",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                str(script),
                "{}:{}".format(self.ssh_target, self.remote_script),
            ],
            timeout=30,
        )
        remote_args = [
            "python3",
            "-u",
            self.remote_script,
            "--interval-ms",
            str(self.interval_ms),
            "--hardware-interval-ms",
            str(self.hardware_interval_ms),
            "--rpc-port",
            str(self.rpc_port),
            "--self-delete",
        ]
        if self.rpc_pid is not None:
            remote_args.extend(["--pid", str(self.rpc_pid)])
        if self.thread_only:
            remote_args.append("--thread-only")
        try:
            self._process = subprocess.Popen(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=8",
                    self.ssh_target,
                    shlex.join(remote_args),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError:
            self._remove_remote_script()
            raise
        self._stdout_thread = threading.Thread(
            target=self._read_stdout, daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            event = self._next_event(lambda item: item.get("type") == "metadata")
        except RuntimeError:
            self.close()
            raise
        self.metadata = event
        return self

    def mark(self, action: str, measurement_id: str) -> None:
        if action not in ("start", "end"):
            raise ValueError("timing marker action must be 'start' or 'end'")
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Nano telemetry has not been started")
        self._token += 1
        token = self._token
        payload = {
            "action": action,
            "measurement_id": measurement_id,
            "token": token,
        }
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()
        self._next_event(
            lambda item: item.get("type") == "marker" and item.get("token") == token
        )

    def close(self) -> dict[str, Any]:
        if self._process is None:
            return self.report()
        if self._process.poll() is None:
            self._token += 1
            token = self._token
            try:
                self._process.stdin.write(
                    json.dumps(
                        {"action": "stop", "measurement_id": None, "token": token}
                    )
                    + "\n"
                )
                self._process.stdin.flush()
                self._next_event(
                    lambda item: item.get("type") == "marker"
                    and item.get("token") == token
                )
                self._process.wait(timeout=5)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                self.cleanup_error = str(error)
                if self._process.poll() is None:
                    self._process.terminate()
                    self._process.wait(timeout=5)
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=5)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        if self._stdout_thread is not None and self._stdout_thread.is_alive():
            self.cleanup_error = "Nano telemetry output reader did not stop"
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            if event is not None:
                self._all_events.append(event)

        self._remove_remote_script()
        return self.report()

    def report(self) -> dict[str, Any]:
        samples = [event for event in self._all_events if event.get("type") == "sample"]
        markers = [event for event in self._all_events if event.get("type") == "marker"]
        shutdown = next(
            (event for event in reversed(self._all_events) if event.get("type") == "shutdown"),
            None,
        )
        return {
            "metadata": self.metadata,
            "markers": markers,
            "samples": samples,
            "shutdown": shutdown,
            "stderr": list(self._stderr_lines),
            "cleanup_error": self.cleanup_error,
        }

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {"type": "protocol_error", "line": line.rstrip()}
            self._events.put(event)
        self._events.put(None)

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr_lines.append(line.rstrip())

    def _next_event(
        self, predicate: Callable[[dict[str, Any]], bool]
    ) -> dict[str, Any]:
        while True:
            try:
                event = self._events.get(timeout=15)
            except queue.Empty as error:
                raise RuntimeError(
                    "Timed out waiting for Nano telemetry; stderr={!r}".format(
                        self._stderr_lines
                    )
                ) from error
            if event is None:
                raise RuntimeError(
                    "Nano telemetry process exited; stderr={!r}".format(
                        self._stderr_lines
                    )
                )
            self._all_events.append(event)
            if event.get("type") == "protocol_error":
                raise RuntimeError(
                    "Invalid telemetry output: {}".format(event["line"])
                )
            if event.get("type") == "error":
                raise RuntimeError(event.get("message", "Nano telemetry failed"))
            if predicate(event):
                return event

    def _run_ssh_tool(self, command: list[str], timeout: int) -> None:
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or str(error)).strip()
            raise RuntimeError(
                "{} failed: {}".format(command[0], detail)
            ) from error

    def _remove_remote_script(self) -> None:
        try:
            self._run_ssh_tool(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=8",
                    self.ssh_target,
                    shlex.join(["rm", "-f", "--", self.remote_script]),
                ],
                timeout=15,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            self.cleanup_error = str(error)
