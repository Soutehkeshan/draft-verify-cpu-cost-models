"""Capture and persist provenance for generated study results."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def capture_provenance(
    repository_root: Path,
    *,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = bool(status.strip())
    if dirty and not allow_dirty:
        raise RuntimeError(
            "Refusing to run with a dirty Git worktree; commit changes or pass --allow-dirty"
        )

    import tvm

    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tvm_version": str(tvm.__version__),
        "python_version": platform.python_version(),
    }


def write_json(path: Path, payload: dict[str, Any], provenance: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {**payload, "provenance": provenance}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(document, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def write_csv(frame, path: Path, provenance: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    write_json(path.with_suffix(".meta.json"), {}, provenance)


def write_companion(path: Path, provenance: dict[str, Any]) -> None:
    write_json(path.with_suffix(".meta.json"), {}, provenance)


def pdf_metadata(provenance: dict[str, Any]) -> dict[str, str]:
    return {
        "Creator": "dvcm",
        "Subject": json.dumps(provenance, sort_keys=True),
        "Keywords": f"git_commit={provenance['git_commit']}",
    }
