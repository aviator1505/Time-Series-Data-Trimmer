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


# ---------------------- FIX 1.3: Nyquist Validation Tests ----------------------
import pytest


def test_butter_lowpass_raises_on_cutoff_above_nyquist():
    """Cutoff frequency must be less than Nyquist (sample_rate / 2)."""
    df = _make_df([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    engine = FilterEngine(sample_rate=100.0)  # Nyquist = 50 Hz

    # Cutoff at Nyquist should raise
    with pytest.raises(ValueError, match="must be less than Nyquist frequency"):
        engine.apply(df, ["ch"], "butter_lowpass", {"cutoff": 50.0, "order": 2})

    # Cutoff above Nyquist should raise
    with pytest.raises(ValueError, match="must be less than Nyquist frequency"):
        engine.apply(df, ["ch"], "butter_lowpass", {"cutoff": 60.0, "order": 2})


def test_butter_lowpass_accepts_valid_cutoff():
    """Cutoff below Nyquist should work fine."""
    df = _make_df([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    engine = FilterEngine(sample_rate=100.0)  # Nyquist = 50 Hz

    # Should not raise
    out = engine.apply(df, ["ch"], "butter_lowpass", {"cutoff": 10.0, "order": 2})
    assert len(out) == len(df)


def test_butter_bandpass_raises_on_cutoff_above_nyquist():
    """Both bandpass cutoffs must be less than Nyquist."""
    df = _make_df([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    engine = FilterEngine(sample_rate=100.0)  # Nyquist = 50 Hz

    # High cutoff at Nyquist should raise
    with pytest.raises(ValueError, match="must be less than Nyquist frequency"):
        engine.apply(df, ["ch"], "butter_bandpass", {"low_cut": 1.0, "high_cut": 50.0, "order": 2})

    # Low cutoff at Nyquist should raise
    with pytest.raises(ValueError, match="must be less than Nyquist frequency"):
        engine.apply(df, ["ch"], "butter_bandpass", {"low_cut": 50.0, "high_cut": 60.0, "order": 2})


def test_butter_bandpass_raises_when_low_exceeds_high():
    """Low cutoff must be less than high cutoff."""
    df = _make_df([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    engine = FilterEngine(sample_rate=100.0)

    with pytest.raises(ValueError, match="must be less than high cutoff"):
        engine.apply(df, ["ch"], "butter_bandpass", {"low_cut": 20.0, "high_cut": 10.0, "order": 2})

    # Equal values should also raise
    with pytest.raises(ValueError, match="must be less than high cutoff"):
        engine.apply(df, ["ch"], "butter_bandpass", {"low_cut": 10.0, "high_cut": 10.0, "order": 2})


# ---------------------- FIX 2.1: Resample NaN Handling Tests ----------------------

def _make_df_with_time(time_vals, data_vals):
    """Create a DataFrame with explicit time values for resampling tests."""
    return pd.DataFrame({
        "normalized_time": np.array(time_vals, dtype=float),
        "ch": np.array(data_vals, dtype=float),
    })


def test_resample_handles_all_valid_values():
    """Resampling should work normally when no NaN values present."""
    df = _make_df_with_time([0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0, 50.0])
    engine = FilterEngine(sample_rate=1.0)
    out = engine.apply(df, ["ch"], "resample", {"target_fs": 2.0})

    # Output should have more samples and no NaN
    assert len(out) > len(df)
    assert not np.any(np.isnan(out["ch"].to_numpy()))


def test_resample_handles_some_nan_values():
    """Resampling should preserve NaN regions in output."""
    # Data with NaN in the middle
    df = _make_df_with_time(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [10.0, np.nan, np.nan, 40.0, 50.0]
    )
    engine = FilterEngine(sample_rate=1.0)
    out = engine.apply(df, ["ch"], "resample", {"target_fs": 2.0})

    # Should have NaN values in the output corresponding to original NaN region
    result = out["ch"].to_numpy()
    assert np.any(np.isnan(result)), "NaN regions should be preserved"

    # First and last values should not be NaN (they were valid in input)
    assert not np.isnan(result[0]), "First value should be valid"
    assert not np.isnan(result[-1]), "Last value should be valid"


def test_resample_handles_all_nan_values():
    """Resampling all-NaN column should produce all-NaN output."""
    df = _make_df_with_time(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [np.nan, np.nan, np.nan, np.nan, np.nan]
    )
    engine = FilterEngine(sample_rate=1.0)
    out = engine.apply(df, ["ch"], "resample", {"target_fs": 2.0})

    # All output values should be NaN
    assert np.all(np.isnan(out["ch"].to_numpy()))


# ---------------------- FIX 2.2: Savgol Numerical Stability Tests ----------------------

def test_savgol_fallback_numerical_stability():
    """Savgol fallback should produce reasonable results even with large indices."""
    # Create data with large time values to test numerical stability
    large_offset = 1000000
    data = np.sin(np.arange(100) * 0.1)  # Simple sinusoidal test data
    df = pd.DataFrame({
        "normalized_time": np.arange(large_offset, large_offset + 100, dtype=float),
        "ch": data,
    })
    engine = FilterEngine(sample_rate=120.0)

    # This should not produce NaN or Inf due to numerical instability
    out = engine.apply(df, ["ch"], "savgol", {"window": 11, "polyorder": 2})
    result = out["ch"].to_numpy()

    assert not np.any(np.isnan(result)), "Savgol should not produce NaN"
    assert not np.any(np.isinf(result)), "Savgol should not produce Inf"
    # Output should be similar to input (smoothed, not drastically different)
    assert np.allclose(result, data, atol=0.5), "Savgol output should be close to input"


def test_savgol_handles_small_data():
    """Savgol should handle data smaller than window size."""
    df = _make_df([1, 2, 3])  # Only 3 points
    engine = FilterEngine(sample_rate=120.0)

    # Window of 11 but only 3 data points - should still work
    out = engine.apply(df, ["ch"], "savgol", {"window": 11, "polyorder": 2})
    result = out["ch"].to_numpy()

    assert len(result) == 3
    assert not np.any(np.isnan(result))
