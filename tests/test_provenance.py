import json

import pandas as pd

from dvcm.provenance import write_csv, write_json


def test_json_and_csv_outputs_record_provenance(tmp_path):
    provenance = {
        "git_commit": "abc123",
        "git_dirty": False,
        "timestamp_utc": "2026-11-12T00:00:00Z",
        "tvm_version": "test",
        "python_version": "3.10.0",
    }

    json_path = tmp_path / "constants.json"
    write_json(json_path, {"value": 1}, provenance)
    assert json.loads(json_path.read_text())["provenance"] == provenance

    csv_path = tmp_path / "evaluation.csv"
    write_csv(pd.DataFrame([{"score": 1}]), csv_path, provenance)
    assert json.loads(tmp_path.joinpath("evaluation.meta.json").read_text())[
        "provenance"
    ] == provenance
