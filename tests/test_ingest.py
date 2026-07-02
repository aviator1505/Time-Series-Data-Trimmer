"""Detection-matrix tests for adaptive ingestion (ingest.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsdt_core.ingest import smart_read, sniff_csv


def _write(tmp_path, text: str, name: str = "data.csv", encoding: str = "utf-8") -> str:
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return str(p)


# ---------------------------------------------------------------------------
# Delimiter / encoding sniffing
# ---------------------------------------------------------------------------

def test_comma_csv_reads_plain(tmp_path):
    path = _write(tmp_path, "normalized_time,gaze_x\n0.0,1.0\n0.01,2.0\n")
    df, report = smart_read(path)
    assert list(df.columns) == ["normalized_time", "gaze_x"]
    assert report.delimiter == ","
    assert len(df) == 2


def test_semicolon_delimiter_detected(tmp_path):
    path = _write(tmp_path, "time_s;gaze_x\n0.0;1.0\n0.01;2.0\n0.02;3.0\n")
    df, report = smart_read(path)
    assert report.delimiter == ";"
    # time_s is detected as the axis, so normalized_time is added
    assert list(df.columns) == ["time_s", "gaze_x", "normalized_time"]


def test_tab_delimiter_detected(tmp_path):
    path = _write(tmp_path, "time_s\tgaze_x\n0.0\t1.0\n0.01\t2.0\n0.02\t3.0\n")
    df, report = smart_read(path)
    assert report.delimiter == "\t"
    assert df["gaze_x"].tolist() == [1.0, 2.0, 3.0]


def test_latin1_encoding_detected(tmp_path):
    path = _write(
        tmp_path,
        "time_s,libellé\n0.0,1.0\n0.01,2.0\n0.02,3.0\n0.03,4.0\n0.04,5.0\n",
        encoding="latin-1",
    )
    df, report = smart_read(path)
    assert "libellé" in df.columns
    assert len(df) == 5


def test_sniff_csv_returns_defaults_for_single_column(tmp_path):
    path = _write(tmp_path, "value\n1\n2\n3\n")
    _encoding, delimiter = sniff_csv(path)
    assert delimiter == ","


# ---------------------------------------------------------------------------
# Time detection and unit conversion
# ---------------------------------------------------------------------------

def test_existing_normalized_time_untouched(tmp_path):
    path = _write(tmp_path, "normalized_time,gaze_x\n5.0,1.0\n5.01,2.0\n")
    df, report = smart_read(path)
    # Values must pass through exactly; no rebasing to zero
    assert df["normalized_time"].tolist() == [5.0, 5.01]
    assert report.created_normalized_time is False
    assert report.time_column == "normalized_time"


def test_epoch_seconds_detected(tmp_path):
    t0 = 1_750_000_000  # unix seconds, 2025
    rows = "\n".join(f"{t0 + i * 0.01},{i}" for i in range(10))
    path = _write(tmp_path, f"timestamp,gaze_x\n{rows}\n")
    df, report = smart_read(path)
    assert report.created_normalized_time
    assert report.time_unit == "s"
    assert df["normalized_time"].iloc[0] == 0.0
    assert df["normalized_time"].iloc[-1] == pytest.approx(0.09, abs=1e-6)


def test_epoch_milliseconds_detected(tmp_path):
    t0 = 1_750_000_000_000  # unix ms
    rows = "\n".join(f"{t0 + i * 10},{i}" for i in range(10))
    path = _write(tmp_path, f"timestamp,gaze_x\n{rows}\n")
    df, report = smart_read(path)
    assert report.time_unit == "ms"
    assert df["normalized_time"].iloc[-1] == pytest.approx(0.09, abs=1e-9)


def test_epoch_microseconds_detected(tmp_path):
    t0 = 1_750_000_000_000_000  # unix us
    rows = "\n".join(f"{t0 + i * 10_000},{i}" for i in range(10))
    path = _write(tmp_path, f"timestamp,gaze_x\n{rows}\n")
    df, report = smart_read(path)
    assert report.time_unit == "us"
    assert df["normalized_time"].iloc[-1] == pytest.approx(0.09, abs=1e-9)


def test_millisecond_name_hint_wins(tmp_path):
    # Small magnitudes (not epoch), but the name says milliseconds
    rows = "\n".join(f"{i * 10},{i}" for i in range(10))
    path = _write(tmp_path, f"time_ms,gaze_x\n{rows}\n")
    df, report = smart_read(path)
    assert report.time_unit == "ms"
    assert df["normalized_time"].iloc[-1] == pytest.approx(0.09, abs=1e-9)


def test_iso_datetime_strings_detected(tmp_path):
    rows = "\n".join(
        f"2026-07-02 10:00:{i:02d}.{i * 10:03d},{i}" for i in range(10)
    )
    path = _write(tmp_path, f"recorded_at_time,gaze_x\n{rows}\n")
    df, report = smart_read(path)
    assert report.time_unit == "datetime"
    assert df["normalized_time"].iloc[0] == 0.0
    assert df["normalized_time"].iloc[1] == pytest.approx(1.01, abs=1e-6)


def test_plain_seconds_left_as_seconds(tmp_path):
    rows = "\n".join(f"{i * 0.02},{i}" for i in range(10))
    path = _write(tmp_path, f"time,gaze_x\n{rows}\n")
    df, report = smart_read(path)
    assert report.time_unit == "s"
    assert df["normalized_time"].iloc[-1] == pytest.approx(0.18, abs=1e-9)


def test_non_monotonic_time_named_column_not_preferred(tmp_path):
    # 'time_to_target' is a signal, not an axis; 'timestamp' is the axis
    rows = "\n".join(f"{1_750_000_000 + i * 0.01},{np.sin(i)}" for i in range(10))
    path = _write(tmp_path, f"timestamp,time_to_target\n{rows}\n")
    df, report = smart_read(path)
    assert report.time_column == "timestamp"


def test_no_time_column_noted(tmp_path):
    path = _write(tmp_path, "gaze_x,gaze_y\n1.0,2.0\n3.0,4.0\n")
    df, report = smart_read(path)
    assert "normalized_time" not in df.columns
    assert report.created_normalized_time is False
    assert any("No time column" in n for n in report.notes)


# ---------------------------------------------------------------------------
# Numeric coercion
# ---------------------------------------------------------------------------

def test_dirty_numeric_column_coerced_with_report(tmp_path):
    path = _write(
        tmp_path,
        "normalized_time,gaze_x\n0.0,1.0\n0.01,2.0\n0.02,bad\n0.03,4.0\n"
        "0.04,5.0\n0.05,6.0\n0.06,7.0\n0.07,8.0\n0.08,9.0\n0.09,10.0\n",
    )
    df, report = smart_read(path)
    assert pd.api.types.is_numeric_dtype(df["gaze_x"])
    assert df["gaze_x"].isna().sum() == 1
    assert report.coerced_columns == {"gaze_x": 1}


def test_mostly_text_column_not_coerced(tmp_path):
    path = _write(
        tmp_path,
        "normalized_time,condition\n0.0,Stand\n0.01,Sit\n0.02,Swivel\n0.03,Stand\n",
    )
    df, report = smart_read(path)
    assert not pd.api.types.is_numeric_dtype(df["condition"])
    assert "condition" not in report.coerced_columns


# ---------------------------------------------------------------------------
# Other formats and integration
# ---------------------------------------------------------------------------

def test_parquet_roundtrip(tmp_path):
    src = pd.DataFrame({
        "normalized_time": np.arange(5) / 100.0,
        "gaze_x": np.arange(5, dtype=float),
    })
    path = str(tmp_path / "data.parquet")
    src.to_parquet(path)
    df, report = smart_read(path)
    assert report.format == "parquet"
    pd.testing.assert_frame_equal(df, src)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        smart_read("/nonexistent/nope.csv")


def test_datamodel_load_csv_uses_smart_ingest(tmp_path):
    """A millisecond time axis must produce a correct sample rate, not 120 Hz."""
    from data_model import DataModel

    rows = "\n".join(f"{i * 10};{i}" for i in range(100))  # 100 Hz in ms
    path = tmp_path / "trial.csv"
    path.write_text(f"time_ms;gaze_x\n{rows}\n")

    dm = DataModel()
    dm.load_csv(str(path))
    assert "normalized_time" in dm.df.columns
    assert dm.sample_rate == pytest.approx(100.0, abs=0.1)
    assert dm.ingest_report is not None
    assert dm.ingest_report.time_unit == "ms"
