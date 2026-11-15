"""scripts/make_report_figures.py: vector figures at column width, without titles."""
import importlib.util
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _figures():
    spec = importlib.util.spec_from_file_location("report_figures_test", ROOT / "scripts" / "make_report_figures.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _learning():
    rows, paired = [], []
    for size in (100, 250, 500, 1000, 2000, 3499):
        for model, level in (("xgb(base)", 0.80), ("xgb(base+analytic)", 0.81)):
            rows.append({"scope": "pooled", "model": model, "train_size": size, "spearman_mean": level,
                         "spearman_bootstrap95_low": level - 0.02, "spearman_bootstrap95_high": level + 0.02})
        paired.append({"scope": "pooled", "metric": "spearman", "train_size": size,
                       "difference_mean_over_seeds": 0.01, "paired_bootstrap_ci_low": 0.002,
                       "paired_bootstrap_ci_high": 0.02})
    return pd.DataFrame(rows), pd.DataFrame(paired)


def _draft():
    return pd.DataFrame([
        {"scope": "pooled", "n_test": 752, "K": k, "draft_method": method, "draft_recall_at_K_top_10_count": hits}
        for k in (32, 64, 128, 256)
        for method, hits in (("xgb_base", 6), ("analytic_fitted", 1), ("analytic_hand_set", 1), ("random", 0))
    ])


def test_random_drafts_are_shown_by_their_expectation_not_one_draw():
    assert _figures().expected_random_hits(32, 752) == pytest.approx(10 * 32 / 752)


@pytest.mark.parametrize("which", ["learning", "draft"])
def test_figures_have_no_titles_and_no_raster_images(which):
    figures = _figures()
    summary, paired = _learning()
    if which == "learning":
        figure = figures.learning_curve_figure({"jetson_nano": summary, "apple_m2": summary},
                                               {"jetson_nano": paired, "apple_m2": paired})
    else:
        figure = figures.draft_recall_figure({"jetson_nano": _draft(), "apple_m2": _draft()})
    assert figure._suptitle is None
    assert all(axis.get_title() == "" for axis in figure.axes)
    assert figure.get_figwidth() == pytest.approx(figures.COLUMN_WIDTH_IN)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="pdf")
    assert b"/Subtype /Image" not in buffer.getvalue()
