import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filter_engine import FilterEngine, available_filters


def _make_df(values):
    return pd.DataFrame(
        {
            "normalized_time": np.arange(len(values), dtype=float),
            "ch": np.array(values, dtype=float),
        }
    )


def test_available_filters_includes_new_entries():
    filters = available_filters()
    assert "moving_rms" in filters
    assert "absolute" in filters
    assert "invert_polarity" in filters
    assert "invert_mean" in filters
    assert "invert_reference" in filters


def test_moving_rms_matches_manual():
    df = _make_df([1, 2, 3, 4, 5])
    engine = FilterEngine(sample_rate=120.0)
    window = 3
    out = engine.apply(df, ["ch"], "moving_rms", {"window": window})

    manual = (
        df["ch"].pow(2)
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .pow(0.5)
    )
    assert np.allclose(out["ch"].to_numpy(), manual.to_numpy())


def test_absolute_filter_handles_negative_values():
    df = _make_df([-2, -1, 0, 1, 2])
    engine = FilterEngine()
    out = engine.apply(df, ["ch"], "absolute", {})
    assert np.array_equal(
        out["ch"].to_numpy(), np.array([2, 1, 0, 1, 2], dtype=float)
    )


def test_invert_polarity_negates_values():
    df = _make_df([-3, -1, 0, 2, 5])
    engine = FilterEngine()
    out = engine.apply(df, ["ch"], "invert_polarity", {})
    expected = np.array([3, 1, 0, -2, -5], dtype=float)
    assert np.array_equal(out["ch"].to_numpy(), expected)


def test_invert_mean_flips_around_mean():
    df = _make_df([1, 2, 3, 4, 5])  # mean = 3
    engine = FilterEngine()
    out = engine.apply(df, ["ch"], "invert_mean", {})
    # 2*3 - [1,2,3,4,5] = [5,4,3,2,1]
    expected = np.array([5, 4, 3, 2, 1], dtype=float)
    assert np.allclose(out["ch"].to_numpy(), expected)


def test_invert_reference_flips_around_reference():
    df = _make_df([0, 5, 10, 15, 20])
    engine = FilterEngine()
    out = engine.apply(df, ["ch"], "invert_reference", {"reference": 10})
    # 2*10 - [0,5,10,15,20] = [20,15,10,5,0]
    expected = np.array([20, 15, 10, 5, 0], dtype=float)
    assert np.array_equal(out["ch"].to_numpy(), expected)
