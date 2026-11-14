"""scripts/02_split.py: a replayed pool reuses its recorded source split."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from dvcm.study import make_splits, validate_splits

ROOT = Path(__file__).resolve().parents[1]


def _splitter():
    spec = importlib.util.spec_from_file_location("split_test", ROOT / "scripts" / "02_split.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(tmp_path: Path, n: int = 60):
    frame = pd.DataFrame({"id": range(n), "sketch_id": [i % 3 for i in range(n)]})
    splits = make_splits(frame, seed=0)
    path = tmp_path / "source_splits.json"
    path.write_text(json.dumps(splits), encoding="utf-8")
    return frame, splits, path


def _replayed(tmp_path: Path, source_path: Path, digest: str | None = None) -> Path:
    data = tmp_path / "replayed"
    data.mkdir()
    digest = digest or hashlib.sha256(source_path.read_bytes()).hexdigest()
    (data / "run_meta.json").write_text(json.dumps(
        {"collection": {"splits": str(source_path), "splits_sha256": digest}}
    ), encoding="utf-8")
    return data


def test_replayed_pool_reuses_the_recorded_source_split(tmp_path):
    splitter = _splitter()
    frame, source, path = _source(tmp_path)
    replayed = frame.iloc[::-1].reset_index(drop=True)  # measured in another order
    assert make_splits(replayed, seed=0)["test"] != source["test"]

    splits = splitter.build_splits(replayed, _replayed(tmp_path, path), seed=0, root=tmp_path)

    for name in ("seed", "train", "validation", "test", "nested_train"):
        assert splits[name] == source[name]
    assert splits["derived_from"]["path"] == "source_splits.json"
    assert splits["derived_from"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_unmodelable_replayed_rows_are_dropped_from_every_role(tmp_path):
    splitter = _splitter()
    frame, source, path = _source(tmp_path)
    dropped = {source["test"][0], source["validation"][0], source["train"][0]}
    measured = frame[~frame["id"].isin(dropped)].reset_index(drop=True)

    splits = splitter.build_splits(measured, _replayed(tmp_path, path), seed=0, root=tmp_path)

    assert splits["test"] == [i for i in source["test"] if i not in dropped]
    assert all(dropped.isdisjoint(ids) for ids in splits["nested_train"].values())
    validate_splits(splits, set(measured["id"]))


def test_changed_source_split_is_refused(tmp_path):
    splitter = _splitter()
    frame, _, path = _source(tmp_path)
    with pytest.raises(ValueError, match="changed since the collection"):
        splitter.build_splits(frame, _replayed(tmp_path, path, digest="0" * 64), seed=0,
                              root=tmp_path)


def test_pool_without_a_recorded_source_draws_a_new_split(tmp_path):
    splitter = _splitter()
    frame, _, _ = _source(tmp_path)
    data = tmp_path / "sampled"
    data.mkdir()
    (data / "run_meta.json").write_text(json.dumps({"collection_complete": True}))

    assert splitter.build_splits(frame, data, seed=0, root=tmp_path) == make_splits(frame, seed=0)
