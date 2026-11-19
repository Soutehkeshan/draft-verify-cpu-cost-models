#!/usr/bin/env python3
"""Check that every number written in Report/*.tex matches a result file.

A number in the text matches when some value in the sources rounds to it at
the number of decimals shown, which also fixes its sign. A number followed
by \\% is compared with 100 times the value, GFLOP/s and GB/s with the value
divided by 1e9, MHz with a kHz value divided by 1e3, and KiB and MiB with a
byte count divided by 1024 and 1024^2. A plain integer must equal a value
exactly. Comments, citations, references, labels, URLs, file names, lengths,
tabular column specifications and affiliations are not checked, and neither
are digits inside names such as A57.

Sources: every CSV and JSON file in results/jetson_nano (models/ excluded)
and results/jetson_nano_confirm, the run, collection and re-timing metadata
and the collection report in data/jetson_nano/, the confirmation run's
metadata and batch log in data/jetson_nano_confirm/, the profiles in
configs/, and the sizes of the split lists. Every .tex file under Report/ is checked, tables included.
Prints each mismatch as path:line and exits 1 if there is any.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIT_SCALES = {
    "": 1.0,
    "%": 100.0,
    "GFLOP/s": 1e-9,
    "GB/s": 1e-9,
    "MHz": 1e-3,
    "KiB": 1.0 / 1024,
    "MiB": 1.0 / 1024**2,
}
# Commands whose arguments are not checked: name -> brace arguments to skip.
# Their optional [...] arguments are always skipped.
SKIPPED_ARGUMENTS = {
    "cite": 1, "citep": 1, "citet": 1, "nocite": 1, "ref": 1, "eqref": 1, "pageref": 1,
    "autoref": 1, "label": 1, "url": 1, "href": 1, "includegraphics": 1, "input": 1,
    "include": 1, "bibliography": 1, "bibliographystyle": 1, "usepackage": 1,
    "documentclass": 1, "multicolumn": 2, "multirow": 2, "cline": 1, "cmidrule": 1,
    "hspace": 1, "vspace": 1, "setlength": 2, "addtolength": 2, "rule": 2, "lipsum": 0,
    "institute": 1, "thanks": 1, "resizebox": 2, "scalebox": 1, "fontsize": 2,
    "raisebox": 1, "renewcommand": 2, "newcommand": 2, "setcounter": 2,
    "addtocounter": 2, "linespread": 1,
}
TABULAR = re.compile(r"\\begin\{(tabular\*|tabularx|tabular|array)\}")
COMMAND = re.compile(r"\\([A-Za-z]+)\*?")
NUMBER = re.compile(
    r"(?P<sign>\$-\$|\$\+\$|\\textminus\s*|−|\+|-)?"
    r"(?P<int>\d{1,3}(?:(?:,|\{,\})\d{3})+|\d+)"
    r"(?P<frac>\.\d+)?"
)
# A unit glued to the digits (2pt, 3mm) is caught by the letters-after-a-number rule;
# only length commands may follow after a space.
LENGTH = re.compile(
    r"\s*\\(?:linewidth|columnwidth|textwidth|textheight|hsize|vsize|baselineskip|"
    r"paperwidth|paperheight)\b"
)
UNIT = re.compile(r"(?:\s|~|\\,|\\ |\\;)*(\\%|GFLOP/s|GB/s|MHz|KiB|MiB)")
NEGATIVE_SIGNS = {"-", "−", "$-$"}
SOURCE_TOKENS = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
# One row per training seed or permutation: thousands of values that a report
# quotes only as means, and that would let a mistyped mean match by chance.
PER_SEED_FILES = {
    "results/jetson_nano/learning_curve.csv",
    "results/jetson_nano/shuffled_label_null_extended.csv",
}
RESULT_DIRS = ("results/jetson_nano", "results/jetson_nano_confirm")


@dataclass(frozen=True)
class Found:
    line: int
    text: str
    value: float
    decimals: int
    unit: str


@dataclass(frozen=True)
class Mismatch:
    path: str
    line: int
    text: str
    unit: str
    context: str


def _group_end(text: str, start: int, opening: str, closing: str) -> int:
    """Index just past the balanced group that opens at text[start]."""
    depth, index = 0, start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return len(text)


def _skip_groups(text: str, index: int, braces: int, options: bool = True) -> int:
    """Index after the optional [...] groups and the next `braces` {...} groups."""
    while True:
        while index < len(text) and text[index] in " \t":
            index += 1
        if options and index < len(text) and text[index] == "[":
            index = _group_end(text, index, "[", "]")
            continue
        if braces and index < len(text) and text[index] == "{":
            index = _group_end(text, index, "{", "}")
            braces -= 1
            continue
        return index


def mask_markup(text: str) -> str:
    """Blank out comments and skipped arguments, keeping every newline and position."""
    chars = list(text)

    def blank(start: int, end: int) -> None:
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "

    for match in re.finditer(r"(?<!\\)%[^\n]*", text):
        blank(match.start(), match.end())
    for match in re.finditer(r"---|--|–|—", text):
        blank(match.start(), match.end())
    masked = "".join(chars)
    for match in TABULAR.finditer(masked):
        braces = 2 if match.group(1) in ("tabular*", "tabularx") else 1
        blank(match.end(), _skip_groups(masked, match.end(), braces))
    for match in COMMAND.finditer(masked):
        braces = SKIPPED_ARGUMENTS.get(match.group(1))
        if braces is not None:
            blank(match.end(), _skip_groups(masked, match.end(), braces))
    return "".join(chars)


def extract_numbers(text: str) -> list[Found]:
    masked = mask_markup(text)
    found = []
    for match in NUMBER.finditer(masked):
        sign = match.group("sign") or ""
        start = match.start()
        if sign in ("+", "-") and start > 0 and re.match(r"[\w)}\]]", masked[start - 1]):
            sign, start = "", match.start("int")  # a hyphen between words or a range
        if not sign and start > 0 and re.match(r"[\w.\\]", masked[start - 1]):
            continue  # digits inside a name (A57, sketch_0) or a decimal tail
        end = match.end()
        if end < len(masked) and re.match(r"\w", masked[end]):
            continue  # a number glued to letters: a length, a unit or a hash
        if LENGTH.match(masked, end):
            continue
        unit_match = UNIT.match(masked, end)
        unit = ""
        if unit_match:
            unit = "%" if unit_match.group(1) == "\\%" else unit_match.group(1)
        digits = re.sub(r"\{,\}|,", "", match.group("int")) + (match.group("frac") or "")
        value = float(digits) * (-1.0 if sign.strip() in NEGATIVE_SIGNS else 1.0)
        found.append(
            Found(
                line=text.count("\n", 0, start) + 1,
                text=text[start:end],
                value=value,
                decimals=len(match.group("frac") or ".") - 1,
                unit=unit,
            )
        )
    return found


class Sources:
    """Numeric values from the result files, searchable at any rounding."""

    def __init__(self, values: list[tuple[float, str]]):
        finite = [(float(v), d) for v, d in values if isinstance(v, (int, float)) and math.isfinite(v)]
        order = sorted(range(len(finite)), key=lambda i: finite[i][0])
        self.values = np.asarray([finite[i][0] for i in order])
        self.descriptions = [finite[i][1] for i in order]

    def __len__(self) -> int:
        return len(self.values)

    def _hits(self, number: Found) -> np.ndarray:
        scale = UNIT_SCALES[number.unit]
        if number.decimals == 0 and not number.unit:
            tolerance = 1e-9 * max(1.0, abs(number.value))
        else:
            tolerance = 0.5 * 10.0 ** (-number.decimals) + 1e-12 * max(1.0, abs(number.value))
        return np.flatnonzero(np.abs(self.values * scale - number.value) <= tolerance)

    def candidates(self, number: Found) -> list[str]:
        """Distinct source columns holding a value that rounds to `number`, in value order."""
        return list(dict.fromkeys(self.descriptions[index] for index in self._hits(number)))

    def distinct_values(self, number: Found) -> int:
        """How many different values round to `number`; copies of one value count once."""
        return len({float(f"{self.values[index]:.10g}") for index in self._hits(number)})

    def match(self, number: Found) -> str | None:
        """Description of a source value that `number` reports, or None."""
        found = self.candidates(number)
        return found[0] if found else None


def _walk(node, prefix: str, out: list) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        out.append((float(node), prefix))
    elif isinstance(node, dict):
        for key, value in node.items():
            _walk(value, f"{prefix}.{key}", out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, f"{prefix}[{index}]", out)


def load_sources(root: Path = ROOT, include_per_seed: bool = False) -> Sources:
    values: list[tuple[float, str]] = []
    paths = [path for directory in RESULT_DIRS for path in sorted((root / directory).glob("*.csv"))]
    for path in paths:
        name = path.relative_to(root).as_posix()
        if name in PER_SEED_FILES and not include_per_seed:
            continue
        frame = pd.read_csv(path)
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                values.extend((float(v), f"{name}:{column}") for v in series.dropna())
            else:
                for cell in series.dropna().astype(str):
                    values.extend((float(t), f"{name}:{column}") for t in SOURCE_TOKENS.findall(cell))
    for path in (path for directory in RESULT_DIRS for path in sorted((root / directory).glob("*.json"))):
        _walk(json.loads(path.read_text(encoding="utf-8")), path.relative_to(root).as_posix(), values)
    for name in ("data/jetson_nano/run_meta.json", "data/jetson_nano/test_retime_meta.json",
                 "data/jetson_nano_confirm/run_meta.json"):
        path = root / name
        if path.is_file():
            _walk(json.loads(path.read_text(encoding="utf-8")), name, values)
    for name in ("data/jetson_nano/collection_batches.jsonl", "data/jetson_nano/test_retime_batches.jsonl",
                 "data/jetson_nano_confirm/collection_batches.jsonl"):
        path = root / name
        if path.is_file():
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                _walk(json.loads(line), f"{name}[{index}]", values)
    report = root / "data" / "jetson_nano" / "collection_report.md"
    if report.is_file():
        values.extend((float(t), report.relative_to(root).as_posix()) for t in SOURCE_TOKENS.findall(report.read_text()))
    for path in sorted((root / "configs").glob("*/*.json")):
        _walk(json.loads(path.read_text(encoding="utf-8")), path.relative_to(root).as_posix(), values)
    for name in ("data/jetson_nano/splits.json",):
        path = root / name
        if path.is_file():
            splits = json.loads(path.read_text(encoding="utf-8"))
            for role in ("train", "validation", "test"):
                values.append((float(len(splits[role])), f"{name}:len({role})"))
            for subset, ids in splits.get("nested_train", {}).items():
                values.append((float(len(ids)), f"{name}:len(nested_train[{subset}])"))
    return Sources(values)


def _display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def check_files(paths: list[Path], sources: Sources, root: Path = ROOT, verbose: bool = False,
                ambiguous: list | None = None):
    """Mismatches and the count of numbers checked; numbers matching several columns go to `ambiguous`."""
    mismatches, checked = [], 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        name = _display(path, root)
        for number in extract_numbers(text):
            checked += 1
            found = sources.candidates(number)
            if not found:
                mismatches.append(
                    Mismatch(name, number.line, number.text, number.unit, lines[number.line - 1].strip()[:110])
                )
                continue
            values = sources.distinct_values(number)
            if values > 1 and ambiguous is not None:
                ambiguous.append((name, number.line, number.text))
            if verbose:
                more = f" ({values} different values round to it)" if values > 1 else " (one value)"
                print(f"ok  {name}:{number.line}: {number.text}{number.unit} <- {', '.join(found[:3])}"
                      f"{' ...' if len(found) > 3 else ''}{more}")
    return mismatches, checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-dir", type=Path, default=ROOT / "Report")
    parser.add_argument("--verbose", action="store_true", help="also print each match and its source")
    parser.add_argument(
        "--all-sources", action="store_true",
        help=f"also accept values from the per-seed files ({', '.join(sorted(PER_SEED_FILES))})",
    )
    args = parser.parse_args()
    paths = sorted(args.report_dir.resolve().rglob("*.tex"))
    sources = load_sources(ROOT, include_per_seed=args.all_sources)
    ambiguous: list = []
    mismatches, checked = check_files(paths, sources, ROOT, args.verbose, ambiguous)
    for item in mismatches:
        print(f"MISMATCH {item.path}:{item.line}: {item.text}{item.unit} -- no result value rounds to it | {item.context}")
    print(
        f"Checked {checked} numbers in {len(paths)} .tex files against {len(sources)} source values"
        f"{'' if args.all_sources else ' (per-seed files excluded; --all-sources includes them)'}: "
        f"{len(mismatches)} mismatches"
    )
    print(
        f"{checked - len(mismatches) - len(ambiguous)} matched a single value; {len(ambiguous)} matched "
        "several different values, so a wrong one there can still pass -- audit them with --verbose"
    )
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
