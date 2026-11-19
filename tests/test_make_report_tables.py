"""scripts/make_report_tables.py: tables copied from result files, rounded but never typed."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _tables():
    spec = importlib.util.spec_from_file_location("report_tables_test", ROOT / "scripts" / "make_report_tables.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_numbers_use_a_math_minus_and_a_plus_only_for_differences():
    tables = _tables()
    assert tables.num(-0.0074, 3) == "$-$0.007"
    assert tables.num(0.0064, 3, signed=True) == "+0.006"
    assert tables.num(0.0064, 3) == "0.006"
    assert tables.num(-1e-17, 2, signed=True) == "0.00"  # never $-$0.00 or +0.00
    assert tables.interval(0.0064, 0.0013, 0.0118, 3, signed=True) == "+0.006 [+0.001, +0.012]"
    assert tables.interval(0.937, 0.923, 0.948, 3) == "0.937 [0.923, 0.948]"


def _paired():
    metrics = ["spearman", "fastest_decile_spearman", "top_10_recall_in_30",
               "top_1_latency_ratio", "top_5_latency_ratio"]
    rows = [{"scope": "pooled", "metric": m, "base_mean_over_seeds": 0.9, "hybrid_mean_over_seeds": 0.91,
             "difference_mean_over_seeds": (i + 1) / 100, "paired_bootstrap_ci_low": i / 100,
             "paired_bootstrap_ci_high": (i + 2) / 100}
            for i, m in enumerate(metrics)]
    rows.append({"scope": "sketch_0", "metric": "spearman", "base_mean_over_seeds": 9.0, "hybrid_mean_over_seeds": 9.0,
                 "difference_mean_over_seeds": 9.0, "paired_bootstrap_ci_low": 9.0, "paired_bootstrap_ci_high": 9.0})
    return pd.DataFrame(rows)


def test_paired_table_has_a_row_per_metric_with_both_models_and_the_difference():
    text = _tables().paired_table(_paired())
    rows = [line for line in text.splitlines() if line.endswith(r"\\") and "&" in line]
    assert rows[0] == r"Metric & XGBoost & + analytical & Difference [95\% CI] \\"
    assert rows[1] == r"Spearman $\rho$ & 0.900 & 0.910 & +0.010 [0.000, +0.020] \\"
    assert len(rows) == 6  # header + five metrics; per-sketch rows are not used
    assert "9.000" not in text


def test_confirmation_table_puts_the_original_and_the_new_gain_side_by_side():
    tables = _tables()
    original = _paired()
    confirm = _paired()
    confirm["metric"] = confirm["metric"].replace({"top_10_recall_in_30": "top_1pct_recall_in_3pct"})
    confirm["difference_mean_over_seeds"] *= -1
    text = tables.confirmation_table(original, confirm)
    rows = [line for line in text.splitlines() if line.endswith(r"\\") and "&" in line]
    assert rows[0] == r"Metric & 752 schedules & 20,000 new schedules \\"
    assert rows[1] == r"Spearman $\rho$ & +0.010 [0.000, +0.020] & $-$0.010 [0.000, +0.020] \\"
    assert rows[3].startswith("Fastest found") and "+0.03 [+0.02, +0.04]" in rows[3]
    assert len(rows) == 6
    assert "9.000" not in text
