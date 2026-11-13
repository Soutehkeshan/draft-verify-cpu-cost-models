"""Load target and workload profiles without embedding hardware values in models."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HARDWARE_CONFIG_DIR = ROOT / "configs" / "hardware"
WORKLOAD_CONFIG_DIR = ROOT / "configs" / "workloads"
PRIMARY_TARGET = "jetson_nano"
LEGACY_TARGET = "apple_m2"
TARGET_IDS = tuple(sorted(path.stem for path in HARDWARE_CONFIG_DIR.glob("*.json")))


@dataclass(frozen=True)
class TargetConfig:
    target_id: str
    name: str
    primary: bool
    architecture: str
    cpu: str
    num_cores: int | None
    simd_width_f32: int | None
    l1_data_cache_bytes: int | None
    l2_cache_bytes: int | None
    vector_registers: int | None
    peak_flops: float | None
    bandwidth_bytes_s: float | None
    compiler_target: dict[str, Any] | None
    compiler_version: str | None
    compiler_options: tuple[str, ...]
    schedule_rules: str | None
    postprocessors: str | None
    thread_settings: dict[str, int]
    data_dir: str
    results_dir: str
    environment: dict[str, Any]
    notes: dict[str, str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TargetConfig:
        values = dict(payload)
        for field in ("compiler_options",):
            values[field] = tuple(values[field])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for field in ("compiler_options",):
            values[field] = list(values[field])
        return values

    def paths(self, root: Path = ROOT) -> tuple[Path, Path]:
        data_dir = Path(self.data_dir)
        results_dir = Path(self.results_dir)
        if data_dir.is_absolute() or results_dir.is_absolute():
            raise ValueError("Target data and results paths must be repository-relative")
        return root / data_dir, root / results_dir

    def feature_settings(self) -> dict[str, int]:
        settings = {
            "vector_lanes": self.simd_width_f32,
            "num_cores": self.num_cores,
            "vector_registers": self.vector_registers,
        }
        missing = [name for name, value in settings.items() if value is None]
        if missing:
            raise ValueError(
                f"Target {self.target_id} needs verified feature settings: "
                f"{', '.join(missing)}"
            )
        return {name: int(value) for name, value in settings.items()}

    def analytic_constants(self) -> dict[str, float | int]:
        values = {
            "peak_flops": self.peak_flops,
            "bandwidth_bytes_s": self.bandwidth_bytes_s,
            "l1_bytes": self.l1_data_cache_bytes,
            "l2_bytes": self.l2_cache_bytes,
            "lane_width": self.simd_width_f32,
            "num_cores": self.num_cores,
            "vector_registers": self.vector_registers,
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(
                f"Target {self.target_id} lacks analytical constants: "
                f"{', '.join(missing)}"
            )
        return {name: value for name, value in values.items() if value is not None}

    def require_benchmarking_profile(self) -> None:
        if (
            self.compiler_target is None
            or not self.schedule_rules
            or not self.postprocessors
        ):
            raise RuntimeError(
                f"Benchmarking is not configured for {self.target_id}; "
                "verify the TVM compiler target, schedule rules, and "
                "postprocessors before enabling collection."
            )

    def make_tvm_target(self):
        self.require_benchmarking_profile()
        import tvm

        return tvm.target.Target(self.compiler_target)


@dataclass(frozen=True)
class WorkloadConfig:
    workload_id: str
    operation: str
    m: int
    n: int
    k: int
    dtype: str
    seed: int
    measurement: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_target_config(target_id: str, root: Path = ROOT) -> TargetConfig:
    path = root / "configs" / "hardware" / f"{target_id}.json"
    if not path.is_file():
        raise ValueError(
            f"Unknown target {target_id!r}; choose one of {', '.join(TARGET_IDS)}"
        )
    with path.open(encoding="utf-8") as source:
        profile = TargetConfig.from_dict(json.load(source))
    if profile.target_id != target_id:
        raise ValueError(f"Target ID in {path} does not match its filename")
    profile.paths(root)
    if profile.num_cores is not None and profile.num_cores < 1:
        raise ValueError("num_cores must be positive when specified")
    for name in (
        "simd_width_f32",
        "l1_data_cache_bytes",
        "l2_cache_bytes",
        "vector_registers",
        "peak_flops",
        "bandwidth_bytes_s",
    ):
        value = getattr(profile, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when specified")
    return profile


def load_workload_config(
    workload_id: str = "matmul_fp32_512", root: Path = ROOT
) -> WorkloadConfig:
    path = root / "configs" / "workloads" / f"{workload_id}.json"
    if not path.is_file():
        raise ValueError(f"Unknown workload {workload_id!r}")
    with path.open(encoding="utf-8") as source:
        workload = WorkloadConfig(**json.load(source))
    if (
        workload.workload_id != workload_id
        or workload.operation != "matrix_multiplication"
        or workload.dtype != "float32"
        or min(workload.m, workload.n, workload.k) < 1
        or workload.seed < 0
        or any(value < 1 for value in workload.measurement.values())
    ):
        raise ValueError(f"Invalid workload configuration in {path}")
    return workload


def add_target_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target",
        choices=TARGET_IDS,
        default=PRIMARY_TARGET,
        help=f"hardware profile (default: {PRIMARY_TARGET})",
    )


def experiment_metadata(
    target: TargetConfig,
    workload: WorkloadConfig,
    data_dir: Path,
) -> dict[str, Any]:
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    dataset: dict[str, str] = {}
    for name in ("pool.parquet", "splits.json", "run_meta.json"):
        path = data_dir / name
        if path.is_file():
            dataset[f"{name}_sha256"] = sha256(path)
    return {
        "target": target.to_dict(),
        "workload": workload.to_dict(),
        "dataset": dataset,
    }
