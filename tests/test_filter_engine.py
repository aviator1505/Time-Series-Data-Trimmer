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
