#!/usr/bin/env python3
"""Write the report's LaTeX tables from the result files.

Reads results/jetson_nano and writes three tabulars to Report/tables/: main.tex
(the five scorers on TEST), paired.tex (XGBoost with and without the analytical
features, seed-paired) and ablation.tex (the fitted analytic model with each
penalty removed). Values are copied and rounded, never recomputed or typed.
XGBoost rows are means over the five training seeds (learning_curve_summary.csv
at full TRAIN, 2,000 TEST resamples); analytic and random rows come from
evaluation.csv. Each file starts with a comment naming its sources and the
commits that produced them; the floats, captions and font size stay in the
report's .tex.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "jetson_nano"
SCORERS = (
    ("random", "Random"),
    ("analytic(hand_set)", "Analytic, hand-set"),
    ("analytic(fitted)", "Analytic, fitted"),
    ("xgb(base)", "XGBoost"),
    ("xgb(base+analytic)", "XGBoost + analytical"),
)
MAIN_METRICS = (
    ("spearman", 3, "spearman_ci_low", "spearman_ci_high"),
    ("fastest_decile_spearman", 3, "fastest_decile_spearman_ci_low", "fastest_decile_spearman_ci_high"),
    ("top_10_recall_in_30", 2, "top_10_recall_ci_low", "top_10_recall_ci_high"),
    ("top_5_latency_ratio", 3, "top_5_ci_low", "top_5_ci_high"),
)
PAIRED_METRICS = (
    ("spearman", r"Spearman $\rho$", 3),
    ("fastest_decile_spearman", r"Fastest-decile $\rho$", 3),
    ("top_10_recall_in_30", "Top-10 in 30", 2),
    ("top_1_latency_ratio", "Top-1 ratio", 3),
    ("top_5_latency_ratio", "Top-5 ratio", 3),
)
ABLATIONS = (
    ("analytic(fitted)", "All penalties"),
    ("without_vector", "No vector"),
    ("without_parallel", "No parallel"),
    ("without_cache", "No cache"),
    ("without_register", "No register"),
)


def num(value: float, digits: int, signed: bool = False) -> str:
    """LaTeX number: a math minus for negatives, a plus for positive differences."""
    rounded = round(float(value), digits) + 0.0
    if rounded == 0:
        return f"{0:.{digits}f}"
    text = f"{abs(rounded):.{digits}f}"
    if rounded < 0:
        return "$-$" + text
    return ("+" if signed else "") + text


def interval(value: float, low: float, high: float, digits: int, signed: bool = False) -> str:
    return f"{num(value, digits, signed)} [{num(low, digits, signed)}, {num(high, digits, signed)}]"


def main_table(evaluation: pd.DataFrame, learning: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Scorer & Spearman $\rho$ & Fastest-decile $\rho$ & Top-10 recall in 30 & Top-5 latency ratio \\",
        r"\hline",
    ]
    scores = evaluation[evaluation["scope"] == "pooled"].set_index("scorer")
    seeds = learning[(learning["scope"] == "pooled") & (learning["subset"].astype(str) == "all")].set_index("model")
    for scorer, name in SCORERS:
        cells = []
        for metric, digits, low, high in MAIN_METRICS:
            if scorer.startswith("xgb"):
                row = seeds.loc[scorer]
                cells.append(interval(row[f"{metric}_mean"], row[f"{metric}_bootstrap95_low"],
                                      row[f"{metric}_bootstrap95_high"], digits))
            else:
                row = scores.loc[scorer]
                cells.append(interval(row[metric], row[low], row[high], digits))
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def paired_table(paired: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}lccc@{}}",
        r"\hline",
        r"Metric & XGBoost & + analytical & Difference [95\% CI] \\",
        r"\hline",
    ]
    pooled = paired[paired["scope"] == "pooled"].set_index("metric")
    for metric, label, digits in PAIRED_METRICS:
        row = pooled.loc[metric]
        lines.append(
            f"{label} & {num(row['base_mean_over_seeds'], digits)} & {num(row['hybrid_mean_over_seeds'], digits)} & "
            + interval(row["difference_mean_over_seeds"], row["paired_bootstrap_ci_low"],
                       row["paired_bootstrap_ci_high"], digits, signed=True)
            + r" \\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def ablation_table(ablations: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{@{}lccc@{}}",
        r"\hline",
        r"Analytic model & $\rho$ & Top-1 & Top-5 \\",
        r"\hline",
    ]
    pooled = ablations[ablations["scope"] == "pooled"].set_index("scorer")
    for scorer, label in ABLATIONS:
        row = pooled.loc[scorer]
        cells = [num(row["spearman"], 3), num(row["top_1_latency_ratio"], 2), num(row["top_5_latency_ratio"], 3)]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _commit(path: Path) -> str:
    meta = path.with_suffix(".meta.json")
    return json.loads(meta.read_text(encoding="utf-8"))["provenance"]["git_commit"][:7]


def _code_state() -> str:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", ".", ":!Report"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return f"{commit}{' with uncommitted changes outside Report/' if changes else ''}"


def write_table(path: Path, body: str, sources: list[Path]) -> None:
    header = [f"% Generated by scripts/make_report_tables.py at {_code_state()}; do not edit by hand."]
    header += [f"% Source: {source.relative_to(ROOT).as_posix()} (commit {_commit(source)})" for source in sources]
    path.write_text("\n".join(header) + "\n" + body, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-dir", type=Path, default=ROOT / "Report")
    args = parser.parse_args()
    out = args.report_dir.resolve() / "tables"
    out.mkdir(parents=True, exist_ok=True)

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(RESULTS / name)

    write_table(out / "main.tex", main_table(read("evaluation.csv"), read("learning_curve_summary.csv")),
                [RESULTS / "evaluation.csv", RESULTS / "learning_curve_summary.csv"])
    write_table(out / "paired.tex", paired_table(read("paired_comparison.csv")), [RESULTS / "paired_comparison.csv"])
    write_table(out / "ablation.tex", ablation_table(read("ablations.csv")), [RESULTS / "ablations.csv"])


if __name__ == "__main__":
    main()
