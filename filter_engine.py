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
                arr = series.to_numpy()
                smooth_window = int(params.get("smooth_window", 0))
                if smooth_window > 1:
                    # Pre-smooth before differentiation to reduce noise amplification
                    arr = pd.Series(arr).rolling(window=smooth_window, min_periods=1, center=True).mean().to_numpy()
                filtered = np.gradient(arr, 1.0 / max(self.sample_rate, 1.0))
            elif filter_type == "integrate":
                dt = 1.0 / max(self.sample_rate, 1.0)
                arr = series.to_numpy()
                valid_mask = np.isfinite(arr)
                if valid_mask.all():
                    # All values valid: standard integration
                    filtered = np.cumsum(arr) * dt
                else:
                    # Handle NaN: integrate valid values, preserve NaN positions
                    # Fill NaN with 0 for integration (no contribution)
                    arr_filled = np.where(valid_mask, arr, 0.0)
                    integrated = np.cumsum(arr_filled) * dt
                    # Restore NaN at original positions only (don't propagate)
                    integrated[~valid_mask] = np.nan
                    filtered = integrated
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
            elif filter_type == "invert_polarity":
                filtered = -series
            elif filter_type == "invert_mean":
                mean_val = series.mean()
                filtered = 2 * mean_val - series
            elif filter_type == "invert_reference":
                ref = float(params.get("reference", 0.0))
                filtered = 2 * ref - series
            elif filter_type == "constant_offset":
                offset = float(params.get("offset", 0.0))
                filtered = series + offset
            else:
                filtered = series
            out.loc[mask, ch] = filtered
        return out

    # ------------------------------------------------------------------
    def _savgol(self, data: np.ndarray, window: int, poly: int) -> np.ndarray:
        # Check for NaN values
        valid_mask = np.isfinite(data)
        has_nan = not valid_mask.all()

        if signal is not None:
            try:
                if has_nan:
                    # SciPy savgol_filter can't handle NaN directly
                    # Apply to valid sections, preserve NaN positions
                    out = np.copy(data)
                    out[valid_mask] = signal.savgol_filter(
                        data[valid_mask], min(window, sum(valid_mask)), poly
                    ) if sum(valid_mask) > poly else data[valid_mask]
                    return out
                return signal.savgol_filter(data, window, poly)
            except Exception:
                pass

        # Polynomial fit fallback: rolling window with poly fit
        half = window // 2
        out = np.copy(data)
        for i in range(len(data)):
            # Skip NaN positions - preserve them in output
            if not valid_mask[i]:
                continue

            lo = max(0, i - half)
            hi = min(len(data), i + half + 1)
            x = np.arange(lo, hi)
            y = data[lo:hi]

            # Only use valid (non-NaN) points for fitting
            window_valid = np.isfinite(y)
            if window_valid.sum() < 2:
                # Not enough valid points for fitting
                out[i] = data[i] if valid_mask[i] else np.nan
                continue

            x_valid = x[window_valid]
            y_valid = y[window_valid]

            try:
                # Center x-coordinates for numerical stability
                x_shifted = x_valid - i
                # Reduce polynomial degree if not enough points
                effective_poly = min(poly, len(x_valid) - 1)
                if effective_poly < 0:
                    out[i] = data[i]
                    continue
                coeffs = np.polyfit(x_shifted, y_valid, deg=effective_poly)
                out[i] = np.polyval(coeffs, 0)  # Evaluate at center (x=0)
            except Exception:
                out[i] = data[i]
        return out

    def _butter_lowpass(self, data: np.ndarray, cutoff: float, order: int) -> np.ndarray:
        nyq = 0.5 * self.sample_rate

        # Validate cutoff against Nyquist frequency
        if cutoff >= nyq:
            raise ValueError(
                f"Cutoff frequency ({cutoff} Hz) must be less than Nyquist frequency ({nyq} Hz). "
                f"Either lower the cutoff or increase the sample rate."
            )

        if cutoff <= 0:
            raise ValueError(f"Cutoff frequency must be positive, got {cutoff} Hz")

        # SciPy is required for accurate Butterworth filtering
        if signal is None:
            raise ImportError(
                "Butterworth lowpass filtering requires SciPy for scientifically accurate results. "
                "The moving average approximation has a fundamentally different frequency response. "
                "Install scipy: pip install scipy"
            )

        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
        return signal.filtfilt(b, a, data)

    def _butter_bandpass(self, data: np.ndarray, low_cut: float, high_cut: float, order: int) -> np.ndarray:
        nyq = 0.5 * self.sample_rate

        # Validate low_cut < high_cut
        if low_cut >= high_cut:
            raise ValueError(
                f"Low cutoff frequency ({low_cut} Hz) must be less than high cutoff frequency ({high_cut} Hz)."
            )

        # Validate both cutoff frequencies against Nyquist
        if low_cut >= nyq:
            raise ValueError(
                f"Low cutoff frequency ({low_cut} Hz) must be less than Nyquist frequency ({nyq} Hz). "
                f"Either lower the cutoff or increase the sample rate."
            )
        if high_cut >= nyq:
            raise ValueError(
                f"High cutoff frequency ({high_cut} Hz) must be less than Nyquist frequency ({nyq} Hz). "
                f"Either lower the cutoff or increase the sample rate."
            )

        # SciPy is required for accurate bandpass filtering
        if signal is None:
            raise ImportError(
                "Butterworth bandpass filtering requires SciPy for scientifically accurate results. "
                "The cascaded moving average approximation produces incorrect frequency response. "
                "Install scipy: pip install scipy"
            )

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
                arr = df[col].to_numpy()
                valid_mask = np.isfinite(arr)
                if valid_mask.all():
                    # All values valid: use normal interpolation
                    out[col] = np.interp(t_new, t_old, arr)
                elif valid_mask.any():
                    # Some NaN values: interpolate only valid values
                    out[col] = np.interp(t_new, t_old[valid_mask], arr[valid_mask])
                    # Mark regions that were originally NaN
                    nan_interp = np.interp(t_new, t_old, (~valid_mask).astype(float))
                    out.loc[nan_interp > 0.5, col] = np.nan
                else:
                    # All NaN input: output all NaN
                    out[col] = np.nan
            else:
                out[col] = df[col].iloc[0]
        if "is_bad_segment" in df.columns:
            out["is_bad_segment"] = np.interp(t_new, t_old, df["is_bad_segment"].astype(float).to_numpy()) > 0.5
        return out


    # ------------------------------------------------------------------
    # Joint/Relative Angle Helpers
    # ------------------------------------------------------------------

    def relative_heading(
        self, df: pd.DataFrame, source_col: str, target_col: str, offset: float = 0.0
    ) -> np.ndarray:
        """Compute relative heading with proper angle wrapping to [-180, 180].

        Args:
            df: DataFrame containing the heading columns
            source_col: Column name for source heading (degrees)
            target_col: Column name for target/reference heading (degrees)
            offset: Additional offset to apply (degrees)

        Returns:
            Array of relative headings in degrees, wrapped to [-180, 180]
        """
        return ((df[source_col] - df[target_col] - offset + 180) % 360) - 180

    def vector_magnitude(
        self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str | None = None
    ) -> np.ndarray:
        """Compute magnitude of 2D or 3D vectors.

        Useful for: speed from velocity components, acceleration magnitude, etc.

        Args:
            df: DataFrame containing the vector component columns
            x_col: Column name for X component
            y_col: Column name for Y component
            z_col: Column name for Z component (optional, for 3D vectors)

        Returns:
            Array of vector magnitudes
        """
        if z_col and z_col in df.columns:
            return np.sqrt(df[x_col] ** 2 + df[y_col] ** 2 + df[z_col] ** 2)
        return np.sqrt(df[x_col] ** 2 + df[y_col] ** 2)

    def quaternion_to_euler(
        self, df: pd.DataFrame, qw_col: str, qx_col: str, qy_col: str, qz_col: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert quaternion columns to Euler angles (yaw, pitch, roll) in degrees.

        Uses ZYX (aerospace) convention.

        Args:
            df: DataFrame containing quaternion columns
            qw_col: Column name for quaternion W component
            qx_col: Column name for quaternion X component
            qy_col: Column name for quaternion Y component
            qz_col: Column name for quaternion Z component

        Returns:
            Tuple of (yaw, pitch, roll) arrays in degrees
        """
        qw = df[qw_col].values
        qx = df[qx_col].values
        qy = df[qy_col].values
        qz = df[qz_col].values

        # Roll (X)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

        # Pitch (Y) - handle gimbal lock
        sinp = 2 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.degrees(np.arcsin(sinp))

        # Yaw (Z)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

        return yaw, pitch, roll

    def quaternion_relative(
        self,
        df: pd.DataFrame,
        parent_qw: str,
        parent_qx: str,
        parent_qy: str,
        parent_qz: str,
        child_qw: str,
        child_qx: str,
        child_qy: str,
        child_qz: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute relative orientation between two quaternion-defined segments.

        Returns the child orientation expressed in the parent frame.
        Result is (yaw, pitch, roll) in degrees representing the rotation from parent to child.

        Uses the formula: relative = parent^-1 * child
        For unit quaternions, q^-1 = (w, -x, -y, -z)

        Args:
            df: DataFrame containing quaternion columns
            parent_qw, parent_qx, parent_qy, parent_qz: Column names for parent quaternion
            child_qw, child_qx, child_qy, child_qz: Column names for child quaternion

        Returns:
            Tuple of (yaw, pitch, roll) arrays in degrees
        """
        pw = df[parent_qw].values
        px = df[parent_qx].values
        py = df[parent_qy].values
        pz = df[parent_qz].values

        cw = df[child_qw].values
        cx = df[child_qx].values
        cy = df[child_qy].values
        cz = df[child_qz].values

        # Parent inverse (for unit quaternions)
        inv_pw, inv_px, inv_py, inv_pz = pw, -px, -py, -pz

        # Quaternion multiplication: inv_parent * child
        rw = inv_pw * cw - inv_px * cx - inv_py * cy - inv_pz * cz
        rx = inv_pw * cx + inv_px * cw + inv_py * cz - inv_pz * cy
        ry = inv_pw * cy - inv_px * cz + inv_py * cw + inv_pz * cx
        rz = inv_pw * cz + inv_px * cy - inv_py * cx + inv_pz * cw

        # Convert result quaternion to Euler
        temp_df = pd.DataFrame({"qw": rw, "qx": rx, "qy": ry, "qz": rz})
        return self.quaternion_to_euler(temp_df, "qw", "qx", "qy", "qz")

    def relative_rotation(
        self,
        df: pd.DataFrame,
        parent_qw: str,
        parent_qx: str,
        parent_qy: str,
        parent_qz: str,
        child_qw: str,
        child_qx: str,
        child_qy: str,
        child_qz: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute relative orientation between two quaternion-defined segments.

        This is an alias for quaternion_relative() providing a more intuitive name.
        Computes: child_rotation * inverse(parent_rotation)

        Args:
            df: DataFrame containing quaternion columns
            parent_qw/qx/qy/qz: Column names for parent segment quaternion
            child_qw/qx/qy/qz: Column names for child segment quaternion

        Returns:
            Tuple of (yaw, pitch, roll) arrays in degrees
        """
        return self.quaternion_relative(
            df,
            parent_qw, parent_qx, parent_qy, parent_qz,
            child_qw, child_qx, child_qy, child_qz,
        )


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
        "invert_polarity",
        "invert_mean",
        "invert_reference",
        "constant_offset",
    ]

