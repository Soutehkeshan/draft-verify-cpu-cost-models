#!/usr/bin/env python3
"""Draw the report's figures from the result files as vector PDFs at column width.

learning_curve.pdf: TEST Spearman of XGBoost with schedule features and with
analytical features at every nested TRAIN size (mean of five seeds, 95%
bootstrap band), above the seed-paired difference with its 95% interval.
draft_recall.pdf: how many of the ten fastest TEST
schedules fall inside a draft of K candidates for the XGBoost and analytic
drafters, against the expectation for a random draft (10 K / n). Figures
carry no titles (the captions do that); the PDF metadata names the sources
and commits.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dvcm.provenance import pdf_metadata

COLUMN_WIDTH_IN = 3.45  # one column of the two-column template, about 250 pt
RESULTS = ROOT / "results" / "jetson_nano"
SIZES = (100, 250, 500, 1000, 2000, 3499)
SIZE_LABELS = ("100", "250", "500", "1k", "2k", "3.5k")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SERIES = {  # colour follows the entity in every figure
    "xgb(base)": ("#2a78d6", "-", "o", "XGBoost"),
    "xgb(base+analytic)": ("#eb6834", "--", "s", "XGBoost + analytical"),
    "analytic_fitted": ("#1baf7a", "-.", "^", "Analytic model"),
}
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.1,
    "pdf.fonttype": 42,
}


def expected_random_hits(k: int, n: int, top: int = 10) -> float:
    """Expected number of the `top` fastest schedules inside a uniformly random draft of k."""
    return top * min(k, n) / n


def _size_axis(axis) -> None:
    axis.set_xscale("log")
    axis.set_xlim(80, 4400)
    axis.set_xticks(SIZES)
    axis.set_xticklabels(SIZE_LABELS)
    axis.minorticks_off()
    axis.grid(True, color=GRID, linewidth=0.4)


def learning_curve_figure(summary: pd.DataFrame, paired: pd.DataFrame):
    with plt.rc_context(STYLE):
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 2.6), sharex=True,
                                             layout="constrained", gridspec_kw={"height_ratios": [1.6, 1.0]})
        summary = summary[summary["scope"] == "pooled"]
        for model in ("xgb(base)", "xgb(base+analytic)"):
            color, line, marker, name = SERIES[model]
            rows = summary[summary["model"] == model].sort_values("train_size")
            top.fill_between(rows["train_size"], rows["spearman_bootstrap95_low"],
                             rows["spearman_bootstrap95_high"], color=color, alpha=0.16, linewidth=0)
            top.plot(rows["train_size"], rows["spearman_mean"], color=color, linestyle=line, marker=marker,
                     markersize=3.2, markeredgecolor=MUTED, markeredgewidth=0.4, label=name)
        differences = paired[(paired["scope"] == "pooled") & (paired["metric"] == "spearman")]
        bottom.axhline(0.0, color=MUTED, linewidth=0.6)
        for row in differences.sort_values("train_size").itertuples(index=False):
            excludes_zero = row.paired_bootstrap_ci_low > 0 or row.paired_bootstrap_ci_high < 0
            bottom.errorbar(row.train_size, row.difference_mean_over_seeds,
                            yerr=[[row.difference_mean_over_seeds - row.paired_bootstrap_ci_low],
                                  [row.paired_bootstrap_ci_high - row.difference_mean_over_seeds]],
                            fmt="o", markersize=3.2, color=INK, ecolor=MUTED, elinewidth=0.7, capsize=1.6,
                            markerfacecolor=INK if excludes_zero else "white", markeredgewidth=0.6)
        for axis in (top, bottom):
            _size_axis(axis)
        bottom.set_xlabel("TRAIN schedules")
        top.set_ylabel(r"Spearman $\rho$")
        bottom.set_ylabel("Difference")
        bottom.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
        bottom.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        figure.align_ylabels([top, bottom])
        handles, labels = top.get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False, handlelength=2.2)
    return figure


def draft_recall_figure(drafts: pd.DataFrame):
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.75), layout="constrained")
        frame = drafts[drafts["scope"] == "pooled"]
        for method in ("xgb_base", "analytic_fitted"):
            color, line, marker, name = SERIES["xgb(base)" if method == "xgb_base" else method]
            rows = frame[frame["draft_method"] == method].sort_values("K")
            axis.plot(rows["K"], rows["draft_recall_at_K_top_10_count"], color=color, linestyle=line,
                      marker=marker, markersize=3.4, markeredgecolor=MUTED, markeredgewidth=0.4, label=name)
        sizes = sorted(frame["K"].unique())
        n_test = int(frame["n_test"].iloc[0])
        axis.plot(sizes, [expected_random_hits(k, n_test) for k in sizes], color=MUTED, linestyle=":",
                  linewidth=1.0, label="Random (expected)")
        axis.set_xscale("log", base=2)
        axis.set_xticks(sizes)
        axis.set_xticklabels([str(k) for k in sizes])
        axis.minorticks_off()
        axis.set_ylim(-0.4, 10.6)
        axis.set_yticks([0, 2, 4, 6, 8, 10])
        axis.grid(True, color=GRID, linewidth=0.4)
        axis.set_xlabel("Draft size $K$")
        axis.set_ylabel("Fastest 10 in draft")
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False, handlelength=2.2,
                      columnspacing=1.2)
    return figure


def _code_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.strip()
    changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", ".", ":!Report"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return {"git_commit": commit, "uncommitted_changes_outside_report": bool(changes)}


def _save(figure, path: Path, sources: list[Path]) -> None:
    provenance = {
        **_code_state(),
        "script": "scripts/make_report_figures.py",
        "sources": {
            source.relative_to(ROOT).as_posix():
            json.loads(source.with_suffix(".meta.json").read_text(encoding="utf-8"))["provenance"]["git_commit"]
            for source in sources
        },
    }
    with plt.rc_context(STYLE):  # fonts resolve at draw time; STYLE embeds TrueType, not Type 3
        figure.savefig(path, format="pdf", metadata={**pdf_metadata(provenance), "CreationDate": None})
    plt.close(figure)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-dir", type=Path, default=ROOT / "Report")
    args = parser.parse_args()
    out = args.report_dir.resolve() / "figures"
    out.mkdir(parents=True, exist_ok=True)

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(RESULTS / name)

    _save(learning_curve_figure(read("learning_curve_summary.csv"), read("learning_curve_paired.csv")),
          out / "learning_curve.pdf", [RESULTS / "learning_curve_summary.csv", RESULTS / "learning_curve_paired.csv"])
    _save(draft_recall_figure(read("draft_verify.csv")), out / "draft_recall.pdf", [RESULTS / "draft_verify.csv"])


if __name__ == "__main__":
    main()
