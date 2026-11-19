"""scripts/check_report_numbers.py: numbers in the report must round from a result value."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _checker():
    spec = importlib.util.spec_from_file_location("report_numbers_test", ROOT / "scripts" / "check_report_numbers.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


TEXT = r"""Gain $+0.006$ [$-$0.002, +0.012] over 3,499 runs and 94.8\% at 12.3~GFLOP/s.
See~\cite{chen2016xgboost} and Fig.~\ref{fig:2}; A57 and L2 hardware. % 999 is a comment
\includegraphics[width=0.48\linewidth]{figures/learning_curve} \vspace{2mm}
Spearman 0.41--0.50, repeats 1-2, and 3{,}499 in math.
\begin{tabular}{lrr}\multicolumn{3}{c}{x}\\[2pt] \end{tabular}
\author{A \institute{London SE10 9LS}}
"""


def test_numbers_are_found_with_sign_unit_and_line_and_markup_is_skipped():
    found = _checker().extract_numbers(TEXT)
    assert [(f.line, f.text, f.value, f.decimals, f.unit) for f in found] == [
        (1, "+0.006", 0.006, 3, ""),
        (1, "$-$0.002", -0.002, 3, ""),
        (1, "+0.012", 0.012, 3, ""),
        (1, "3,499", 3499.0, 0, ""),
        (1, "94.8", 94.8, 1, "%"),
        (1, "12.3", 12.3, 1, "GFLOP/s"),
        (4, "0.41", 0.41, 2, ""),
        (4, "0.50", 0.5, 2, ""),
        (4, "1", 1.0, 0, ""),
        (4, "2", 2.0, 0, ""),
        (4, "3{,}499", 3499.0, 0, ""),
    ]


def test_a_number_matches_when_a_result_value_rounds_to_it():
    checker = _checker()
    sources = checker.Sources([(0.00641, "a"), (-0.0017, "b"), (0.948, "c"), (1.2259e10, "d"),
                               (3499.0, "e"), (0.2128, "f"), (2.6, "g")])
    found = {f.text: f for f in checker.extract_numbers(
        r"+0.006, $-$0.002, $-$0.006, 94.8\%, 12.3~GFLOP/s, 3,499, 21\%, 3 runs, 0.97")}
    assert sources.match(found["+0.006"]) == "a"
    assert sources.match(found["$-$0.002"]) == "b"
    assert sources.match(found["$-$0.006"]) is None  # the sign must agree
    assert sources.match(found["94.8"]) == "c"  # percent of 0.948
    assert sources.match(found["12.3"]) == "d"  # giga of 1.2259e10
    assert sources.match(found["3,499"]) == "e"
    assert sources.match(found["21"]) == "f"  # a rounded percentage
    assert sources.match(found["3"]) is None  # a plain integer must be exact; 2.6 does not count
    assert sources.match(found["0.97"]) is None  # 0.948 rounds to 0.95, not 0.97


def test_mismatches_are_reported_with_file_and_line(tmp_path):
    checker = _checker()
    report = tmp_path / "Report"
    (report / "tables").mkdir(parents=True)
    (report / "main.tex").write_text("We saw 0.937 and 0.555.\n\\input{tables/t}\n")
    (report / "tables" / "t.tex").write_text("% generated\n1.250 & 7 \\\\\n")
    sources = checker.Sources([(0.93701, "x"), (1.2504, "y"), (7.0, "z")])
    mismatches, checked = checker.check_files(sorted(report.rglob("*.tex")), sources, root=tmp_path)
    assert checked == 4
    assert [(m.path, m.line, m.text) for m in mismatches] == [("Report/main.tex", 1, "0.555")]


def test_a_word_after_a_number_is_not_mistaken_for_a_unit():
    found = _checker().extract_numbers(r"a gain of 0.006 in Spearman, 2 mm, 0.5\columnwidth, 3pt")
    assert [f.text for f in found] == ["0.006", "2"]


def test_per_seed_files_are_left_out_unless_asked_for(tmp_path):
    checker = _checker()
    results = tmp_path / "results" / "jetson_nano"
    results.mkdir(parents=True)
    (results / "evaluation.csv").write_text("scorer,spearman\nx,0.937\n")
    (results / "learning_curve.csv").write_text("seed,spearman\n0,0.9381\n")  # one seed's value
    typo = checker.extract_numbers("0.938")[0]
    assert checker.load_sources(tmp_path).match(typo) is None
    assert checker.load_sources(tmp_path, include_per_seed=True).match(typo) == "results/jetson_nano/learning_curve.csv:spearman"


def test_ambiguity_counts_distinct_values_not_copies_of_one_value():
    checker = _checker()
    sources = checker.Sources([(0.9371, "a.csv:x"), (0.9374, "b.csv:y"), (0.9374, "c.csv:y"), (0.512, "c.csv:z"),
                               (0.512, "d.csv:z")])
    common, rare = checker.extract_numbers("0.937 and 0.512")
    assert sources.candidates(common) == ["a.csv:x", "b.csv:y", "c.csv:y"]
    assert sources.distinct_values(common) == 2  # two different quantities round to 0.937
    assert sources.distinct_values(rare) == 1  # one quantity, copied into two files


def test_confirmation_results_are_sources_and_their_learning_curve_is_not_per_seed(tmp_path):
    checker = _checker()
    (tmp_path / "results" / "jetson_nano").mkdir(parents=True)
    (tmp_path / "results" / "jetson_nano" / "learning_curve.csv").write_text("seed,spearman\n0,0.9381\n")
    confirm = tmp_path / "results" / "jetson_nano_confirm"
    confirm.mkdir(parents=True)
    (confirm / "paired.csv").write_text("metric,difference\nspearman,0.00512\n")
    (confirm / "learning_curve.csv").write_text("subset,difference\n500,0.01741\n")  # one row per size, not per seed
    sources = checker.load_sources(tmp_path)
    gain, curve, per_seed = checker.extract_numbers("+0.005, 0.017 and 0.938")
    assert sources.match(gain) == "results/jetson_nano_confirm/paired.csv:difference"
    assert sources.match(curve) == "results/jetson_nano_confirm/learning_curve.csv:difference"
    assert sources.match(per_seed) is None
