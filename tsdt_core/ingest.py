"""Adaptive tabular ingestion.

Reads CSV/TSV (with delimiter + encoding sniffing), Parquet, and Feather
into a DataFrame ready for the DataModel:

- object columns that are mostly numeric are coerced, with a per-column
  count of values lost to coercion
- the time column is detected by name and monotonicity; its unit is
  inferred (epoch s/ms/us/ns, ISO datetimes, plain seconds, name hints
  like `_ms`) and converted into a `normalized_time` column of relative
  seconds when one does not already exist
- everything that was detected, converted, or skipped is recorded in an
  IngestReport for display to the user

A file that already contains `normalized_time` is passed through with no
time handling at all, preserving the app's historical behavior.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd
from charset_normalizer import from_bytes
from pydantic import BaseModel, Field

SNIFF_BYTES = 65536
CANDIDATE_DELIMITERS = ",;\t|"
# Fraction of non-null values that must convert for an object column to
# be coerced to numeric.
COERCE_THRESHOLD = 0.9
# Fraction of values that must parse as datetimes for a column to be
# treated as a datetime time axis.
DATETIME_THRESHOLD = 0.95

TIME_NAME_KEYWORDS = (
    "time", "timestamp", "datetime", "clock", "sec", "millis", "micros",
)

# Median-magnitude thresholds for epoch detection. Unix seconds are ~1.7e9
# in 2026; each unit is 1000x the previous.
EPOCH_THRESHOLDS = (
    (1e16, "ns", 1e-9),
    (1e13, "us", 1e-6),
    (1e10, "ms", 1e-3),
    (1e8, "s", 1.0),
)

NAME_UNIT_HINTS = (
    (("_ns", "nanos"), "ns", 1e-9),
    (("_us", "micros"), "us", 1e-6),
    (("_ms", "millis", "msec"), "ms", 1e-3),
)


class IngestReport(BaseModel):
    """What smart_read detected and did. Shown to the user after load."""

    source: str = ""
    format: str = "csv"
    encoding: str | None = None
    delimiter: str | None = None
    time_column: str | None = None
    time_unit: str | None = None  # s / ms / us / ns / datetime
    created_normalized_time: bool = False
    coerced_columns: dict[str, int] = Field(default_factory=dict)  # col -> values lost
    notes: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.delimiter and self.delimiter != ",":
            parts.append(f"delimiter {self.delimiter!r}")
        if self.encoding and self.encoding.lower() not in ("ascii", "utf-8", "utf_8"):
            parts.append(f"encoding {self.encoding}")
        if self.created_normalized_time:
            parts.append(f"time from '{self.time_column}' ({self.time_unit})")
        if self.coerced_columns:
            parts.append(f"{len(self.coerced_columns)} column(s) coerced to numeric")
        return "; ".join(parts)


def sniff_csv(path: str) -> tuple[str, str]:
    """Detect (encoding, delimiter) from the head of the file."""
    with open(path, "rb") as f:
        raw = f.read(SNIFF_BYTES)
    best = from_bytes(raw).best()
    encoding = best.encoding if best is not None else "utf-8"
    sample = raw.decode(encoding, errors="replace")
    # Sniff on complete lines only: a truncated last line skews the sniffer.
    sample = sample[: sample.rfind("\n") + 1] or sample
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=CANDIDATE_DELIMITERS)
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    return encoding, delimiter


def smart_read(path: str) -> tuple[pd.DataFrame, IngestReport]:
    """Read any supported tabular file and prepare it for the DataModel."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    report = IngestReport(source=os.path.basename(path))
    ext = os.path.splitext(path)[1].lower()

    if ext == ".parquet":
        df = pd.read_parquet(path)
        report.format = "parquet"
    elif ext in (".feather", ".arrow"):
        df = pd.read_feather(path)
        report.format = "feather"
    else:
        encoding, delimiter = sniff_csv(path)
        report.encoding = encoding
        report.delimiter = delimiter
        df = pd.read_csv(path, sep=delimiter, encoding=encoding)

    # Normalize NaN spellings (historical behavior of DataModel.load_csv)
    df = df.replace({"": np.nan, "nan": np.nan, "NaN": np.nan})

    df = _coerce_numeric_columns(df, report)
    df = _build_normalized_time(df, report)
    return df, report


def _coerce_numeric_columns(df: pd.DataFrame, report: IngestReport) -> pd.DataFrame:
    """Convert mostly-numeric string columns to numbers, recording losses."""
    for col in df.columns:
        # Skip anything already typed (covers both the legacy object dtype
        # and the pandas>=3 default string dtype for text columns).
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            continue
        non_null = df[col].notna().sum()
        if non_null == 0:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        ok = converted.notna().sum()
        if ok / non_null >= COERCE_THRESHOLD:
            lost = int(non_null - ok)
            df[col] = converted
            if lost:
                report.coerced_columns[col] = lost
                report.notes.append(
                    f"Column '{col}': {lost} non-numeric value(s) became NaN"
                )
    return df


def _build_normalized_time(df: pd.DataFrame, report: IngestReport) -> pd.DataFrame:
    """Detect the time axis and create normalized_time (relative seconds)."""
    if "normalized_time" in df.columns:
        report.time_column = "normalized_time"
        report.time_unit = "s"
        return df

    candidate = _detect_time_column(df)
    if candidate is None:
        report.notes.append("No time column detected; sample-rate fallback applies")
        return df

    col = df[candidate]
    seconds: pd.Series | None = None
    unit: str | None = None

    if pd.api.types.is_datetime64_any_dtype(col):
        seconds, unit = _seconds_from_datetime(col), "datetime"
    elif pd.api.types.is_numeric_dtype(col):
        seconds, unit = _seconds_from_numeric(candidate, col)
    else:
        parsed = pd.to_datetime(col, errors="coerce", format="mixed")
        non_null = col.notna().sum()
        if non_null and parsed.notna().sum() / non_null >= DATETIME_THRESHOLD:
            seconds, unit = _seconds_from_datetime(parsed), "datetime"

    if seconds is None:
        report.notes.append(
            f"Time candidate '{candidate}' has an unrecognized format; left untouched"
        )
        return df

    df = df.copy()
    df["normalized_time"] = seconds
    report.time_column = candidate
    report.time_unit = unit
    report.created_normalized_time = True
    return df


def _detect_time_column(df: pd.DataFrame) -> str | None:
    """Pick the best time-axis candidate by name, preferring monotonic ones."""
    named = [
        c for c in df.columns
        if any(k in str(c).lower() for k in TIME_NAME_KEYWORDS)
    ]
    if not named:
        return None
    monotonic = []
    for c in named:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_datetime64_any_dtype(col):
            vals = col.dropna()
            if len(vals) >= 2 and vals.is_monotonic_increasing:
                monotonic.append(c)
    if monotonic:
        return monotonic[0]
    return named[0]


def _seconds_from_datetime(col: pd.Series) -> pd.Series:
    first = col.dropna().iloc[0]
    return (col - first).dt.total_seconds()


def _seconds_from_numeric(name: str, col: pd.Series) -> tuple[pd.Series | None, str | None]:
    """Convert a numeric time column to relative seconds.

    Unit resolution order: explicit name hint (`_ms`, `micros`, ...), then
    epoch-magnitude detection, then plain seconds as-is.
    """
    vals = col.dropna()
    if len(vals) < 2:
        return None, None

    lname = str(name).lower()
    factor: float | None = None
    unit: str | None = None
    for suffixes, hint_unit, hint_factor in NAME_UNIT_HINTS:
        if any(s in lname for s in suffixes):
            unit, factor = hint_unit, hint_factor
            break
    if factor is None:
        median = float(np.median(np.abs(vals)))
        for threshold, epoch_unit, epoch_factor in EPOCH_THRESHOLDS:
            if median > threshold:
                unit, factor = epoch_unit, epoch_factor
                break
    if factor is None:
        unit, factor = "s", 1.0

    first = float(vals.iloc[0])
    return (col - first) * factor, unit
