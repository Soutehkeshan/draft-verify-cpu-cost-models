"""TVM RPC execution helpers for cross-compiled CPU modules."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
import tvm


@dataclass(frozen=True)
class RemoteLibrary:
    module: Any
    export_time_s: float
    upload_time_s: float
    link_load_time_s: float


def export_rpc_archive(library: Any, archive_path: str | Path) -> float:
    """Export a TVM module as a target-linkable archive."""
    previous_copyfile_disable = os.environ.get("COPYFILE_DISABLE")
    if sys.platform == "darwin":
        os.environ["COPYFILE_DISABLE"] = "1"
    try:
        start = time.perf_counter()
        library.export_library(str(archive_path))
        return time.perf_counter() - start
    finally:
        if sys.platform == "darwin":
            if previous_copyfile_disable is None:
                os.environ.pop("COPYFILE_DISABLE", None)
            else:
                os.environ["COPYFILE_DISABLE"] = previous_copyfile_disable


class RPCExecutor:
    """Own a TVM RPC session and reusable remote CPU inputs."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self.device = session.cpu()
        self._module_count = 0
        self._inputs: dict[tuple[int, int], tuple[Any, Any, Any, np.ndarray]] = {}
        self.input_prepare_time_s = 0.0
        self.input_upload_time_s = 0.0

    @classmethod
    def connect(cls, host: str, port: int = 9090) -> RPCExecutor:
        return cls(tvm.rpc.connect(host, port))

    def load_module(self, library: Any) -> RemoteLibrary:
        """Export, upload, and load a module, keeping each phase duration separate."""
        self._module_count += 1
        remote_name = f"dvcm_module_{self._module_count:06d}.tar"
        with tempfile.TemporaryDirectory(prefix="dvcm-rpc-") as temp_dir:
            archive_path = Path(temp_dir) / remote_name
            export_time_s = export_rpc_archive(library, archive_path)

            start = time.perf_counter()
            self.session.upload(str(archive_path))
            upload_time_s = time.perf_counter() - start

            start = time.perf_counter()
            module = self.session.load_module(remote_name)
            link_load_time_s = time.perf_counter() - start

            self.session.remove(remote_name)
            self.session.remove(f"{remote_name}.so")

        return RemoteLibrary(
            module=module,
            export_time_s=export_time_s,
            upload_time_s=upload_time_s,
            link_load_time_s=link_load_time_s,
        )

    def prepare_inputs(
        self, n: int, seed: int
    ) -> tuple[Any, Any, Any, np.ndarray]:
        """Allocate and cache seeded matmul inputs on the remote CPU."""
        key = (n, seed)
        if key not in self._inputs:
            start = time.perf_counter()
            rng = np.random.default_rng(seed)
            a_np = rng.random((n, n), dtype=np.float32)
            b_np = rng.random((n, n), dtype=np.float32)
            reference = a_np @ b_np
            self.input_prepare_time_s += time.perf_counter() - start

            to_device = getattr(tvm.runtime, "tensor", None) or tvm.nd.array
            start = time.perf_counter()
            a = to_device(a_np, self.device)
            b = to_device(b_np, self.device)
            c = to_device(np.zeros((n, n), dtype=np.float32), self.device)
            self.input_upload_time_s += time.perf_counter() - start
            self._inputs[key] = (a, b, c, reference)
        return self._inputs[key]
