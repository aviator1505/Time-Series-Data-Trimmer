"""Filter engine for time-series operations."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import signal  # type: ignore
except Exception:  # pragma: no cover - fallback when scipy missing
    signal = None


class FilterEngine:
    """Collection of filtering utilities applied to pandas DataFrames."""

    def __init__(self, sample_rate: float = 120.0) -> None:
        self.sample_rate = sample_rate

    def set_sample_rate(self, fs: float) -> None:
        self.sample_rate = float(fs)

    # ------------------------------------------------------------------
    def apply(self, df: pd.DataFrame, channels: Iterable[str], filter_type: str, params: Dict, selection: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        channels = list(channels)
        if not channels:
            return df
        out = df.copy()
        mask = np.ones(len(out), dtype=bool)
        if selection:
            start, end = selection
            mask = (out["normalized_time"] >= start) & (out["normalized_time"] <= end)
        for ch in channels:
            if ch not in out.columns:
                continue
            series = out.loc[mask, ch]
            if filter_type == "moving_average":
                window = int(params.get("window", 5))
                filtered = series.rolling(window=window, min_periods=1, center=True).mean()
            elif filter_type == "median":
                window = int(params.get("window", 5))
                filtered = series.rolling(window=window, center=True, min_periods=1).median()
            elif filter_type == "savgol":
                win = int(params.get("window", 11))
                poly = int(params.get("polyorder", 2))
                if win % 2 == 0:
                    win += 1
                filtered = self._savgol(series.to_numpy(), win, poly)
            elif filter_type == "butter_lowpass":
                cutoff = float(params.get("cutoff", 6.0))
                order = int(params.get("order", 2))
                filtered = self._butter_lowpass(series.to_numpy(), cutoff, order)
            elif filter_type == "butter_bandpass":
                lo = float(params.get("low_cut", 0.5))
                hi = float(params.get("high_cut", 10.0))
                order = int(params.get("order", 2))
                filtered = self._butter_bandpass(series.to_numpy(), lo, hi, order)
            elif filter_type == "detrend":
                filtered = self._detrend(series.to_numpy())
            elif filter_type == "derivative":
                filtered = np.gradient(series.to_numpy(), 1.0 / max(self.sample_rate, 1.0))
            elif filter_type == "integrate":
                dt = 1.0 / max(self.sample_rate, 1.0)
                filtered = np.cumsum(series.to_numpy()) * dt
            elif filter_type == "normalize_zscore":
                arr = series.to_numpy()
                std = np.nanstd(arr) or 1.0
                filtered = (arr - np.nanmean(arr)) / std
            elif filter_type == "normalize_percent":
                arr = series.to_numpy()
                m = np.nanmax(np.abs(arr)) or 1.0
                filtered = arr / m * 100.0
            elif filter_type == "moving_rms":
                window = max(1, int(params.get("window", 5)))
                sq = series.pow(2)
                filtered = sq.rolling(window=window, min_periods=1, center=True).mean().pow(0.5)
            elif filter_type in ("abs", "absolute"):
                filtered = series.abs()
            elif filter_type == "resample":
                target_fs = float(params.get("target_fs", self.sample_rate))
                out = self._resample(out, target_fs)
                self.sample_rate = target_fs
                return out
            elif filter_type == "interpolate":
                method = params.get("method", "linear")
                filtered = series.interpolate(method=method, limit_direction="both")
            else:
                filtered = series
            out.loc[mask, ch] = filtered
        return out

    # ------------------------------------------------------------------
    def _savgol(self, data: np.ndarray, window: int, poly: int) -> np.ndarray:
        if signal is not None:
            try:
                return signal.savgol_filter(data, window, poly)
            except Exception:
                pass
        # simple polynomial fit fallback: rolling window with poly fit
        half = window // 2
        out = np.copy(data)
        for i in range(len(data)):
            lo = max(0, i - half)
            hi = min(len(data), i + half + 1)
            x = np.arange(lo, hi)
            y = data[lo:hi]
            try:
                coeffs = np.polyfit(x, y, deg=min(poly, len(x) - 1))
                out[i] = np.polyval(coeffs, i)
            except Exception:
                out[i] = data[i]
        return out

    def _butter_lowpass(self, data: np.ndarray, cutoff: float, order: int) -> np.ndarray:
        if signal is None or cutoff <= 0:
            # fallback to moving average if SciPy is unavailable
            window = max(3, int(self.sample_rate / max(cutoff, 1)))
            return pd.Series(data).rolling(window=window, min_periods=1, center=True).mean().to_numpy()
        nyq = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
        return signal.filtfilt(b, a, data)

    def _butter_bandpass(self, data: np.ndarray, low_cut: float, high_cut: float, order: int) -> np.ndarray:
        if signal is None:
            # simple detrend + lowpass fallback
            data = self._detrend(data)
            return self._butter_lowpass(data, high_cut, order)
        nyq = 0.5 * self.sample_rate
        low = low_cut / nyq
        high = high_cut / nyq
        b, a = signal.butter(order, [low, high], btype="band")
        return signal.filtfilt(b, a, data)

    def _detrend(self, data: np.ndarray) -> np.ndarray:
        if signal is not None:
            try:
                return signal.detrend(data)
            except Exception:
                pass
        # simple linear detrend fallback
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend = np.polyval(coeffs, x)
        return data - trend

    def _resample(self, df: pd.DataFrame, target_fs: float) -> pd.DataFrame:
        if "normalized_time" not in df.columns:
            return df
        t_old = df["normalized_time"].to_numpy()
        if len(t_old) < 2:
            return df
        duration = t_old[-1]
        n_new = int(duration * target_fs)
        if n_new <= 1:
            return df
        t_new = np.arange(n_new) / target_fs
        out = pd.DataFrame()
        out["normalized_time"] = t_new
        for col in df.columns:
            if col == "normalized_time":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                out[col] = np.interp(t_new, t_old, df[col].to_numpy())
            else:
                out[col] = df[col].iloc[0]
        if "is_bad_segment" in df.columns:
            out["is_bad_segment"] = np.interp(t_new, t_old, df["is_bad_segment"].astype(float).to_numpy()) > 0.5
        return out


def available_filters() -> List[str]:
    return [
        "moving_average",
        "median",
        "savgol",
        "butter_lowpass",
        "butter_bandpass",
        "detrend",
        "resample",
        "interpolate",
        "derivative",
        "integrate",
        "normalize_zscore",
        "normalize_percent",
        "moving_rms",
        "absolute",
    ]

