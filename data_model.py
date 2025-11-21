"""Core data model for time-series cleaning and annotation.

This module exposes DataModel which wraps a pandas DataFrame and
provides undo/redo, deletion with time collapse, masked segments,
annotation persistence, and operation history. It is UI-agnostic so it
can be reused in tests or other front-ends.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt6 import QtCore


@dataclass
class AnnotationSegment:
    start: float
    end: float
    label: str
    track: str = "default"
    color: str = "#4e79a7"
    id: int = field(default_factory=int)


@dataclass
class OperationRecord:
    description: str
    params: Dict
    start: float
    end: float


class DataModel(QtCore.QObject):
    """Backend for time-series data with undo/redo and annotation support."""

    dataChanged = QtCore.pyqtSignal()
    annotationsChanged = QtCore.pyqtSignal()
    statusMessage = QtCore.pyqtSignal(str)
    historyChanged = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.time_columns: List[str] = []
        self.metadata_columns: List[str] = []
        self.signal_columns: List[str] = []
        self.annotations: List[AnnotationSegment] = []
        self.deletions: List[Tuple[float, float]] = []
        self.history: List[OperationRecord] = []
        self.sample_rate: float = 120.0
        self._undo_stack: List[Tuple[pd.DataFrame, List[AnnotationSegment], List[Tuple[float, float]], List[OperationRecord]]] = []
        self._redo_stack: List[Tuple[pd.DataFrame, List[AnnotationSegment], List[Tuple[float, float]], List[OperationRecord]]] = []
        self._id_counter: int = 1

    # ------------------------------------------------------------------
    # Loading and classification
    # ------------------------------------------------------------------
    def load_csv(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        # Normalize NaNs
        df = df.replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
        self.original_df = df.copy()
        self.df = df.copy()
        self._classify_columns(df)
        self._ensure_bad_mask()
        self.deletions.clear()
        self.annotations.clear()
        self.history.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._id_counter = 1
        self.sample_rate = self._infer_sample_rate()
        self.dataChanged.emit()
        self.statusMessage.emit(f"Loaded {os.path.basename(path)}")

    def _classify_columns(self, df: pd.DataFrame) -> None:
        time_candidates = [c for c in df.columns if "time" in c.lower()]
        if "normalized_time" in df.columns:
            self.time_columns = ["normalized_time"]
        elif time_candidates:
            self.time_columns = [time_candidates[0]]
        else:
            self.time_columns = []
        metadata_cols: List[str] = []
        signal_cols: List[str] = []
        for col in df.columns:
            if col in self.time_columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                signal_cols.append(col)
            else:
                metadata_cols.append(col)
        # heuristic grouping later on
        self.metadata_columns = metadata_cols
        self.signal_columns = signal_cols

    def _ensure_bad_mask(self) -> None:
        if self.df is None:
            return
        if "is_bad_segment" not in self.df.columns:
            self.df["is_bad_segment"] = False
        if "normalized_time" not in self.df.columns:
            # fabricate time axis based on sample_rate
            n = len(self.df)
            self.df["normalized_time"] = np.arange(n) / self.sample_rate
            self.time_columns.insert(0, "normalized_time")

    def _infer_sample_rate(self) -> float:
        if self.df is None or "normalized_time" not in self.df.columns:
            return 120.0
        t = self.df["normalized_time"].values
        if len(t) < 2:
            return 120.0
        diffs = np.diff(t)
        median_dt = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 0
        if median_dt <= 0:
            return 120.0
        return float(np.round(1.0 / median_dt, 2))

    # ------------------------------------------------------------------
    # Undo / redo helpers
    # ------------------------------------------------------------------
    def _push_state(self) -> None:
        if self.df is None:
            return
        self._undo_stack.append(
            (self.df.copy(), list(self.annotations), list(self.deletions), list(self.history))
        )
        self._redo_stack.clear()

    def undo(self) -> None:
        if not self._undo_stack:
            self.statusMessage.emit("Nothing to undo")
            return
        if self.df is not None:
            self._redo_stack.append(
                (self.df.copy(), list(self.annotations), list(self.deletions), list(self.history))
            )
        self.df, self.annotations, self.deletions, self.history = self._undo_stack.pop()
        self.dataChanged.emit()
        self.annotationsChanged.emit()
        self.historyChanged.emit()
        self.statusMessage.emit("Undo")

    def redo(self) -> None:
        if not self._redo_stack:
            self.statusMessage.emit("Nothing to redo")
            return
        if self.df is not None:
            self._undo_stack.append(
                (self.df.copy(), list(self.annotations), list(self.deletions), list(self.history))
            )
        self.df, self.annotations, self.deletions, self.history = self._redo_stack.pop()
        self.dataChanged.emit()
        self.annotationsChanged.emit()
        self.historyChanged.emit()
        self.statusMessage.emit("Redo")

    # ------------------------------------------------------------------
    # Core editing operations
    # ------------------------------------------------------------------
    def delete_segment(self, start: float, end: float) -> None:
        if self.df is None or start >= end:
            self.statusMessage.emit("Invalid delete range")
            return
        self._push_state()
        mask = (self.df["normalized_time"] < start) | (self.df["normalized_time"] > end)
        new_df = self.df.loc[mask].reset_index(drop=True)
        n = len(new_df)
        new_df["normalized_time"] = np.arange(n) / self.sample_rate
        self.df = new_df
        self.deletions.append((start, end))
        self.history.append(OperationRecord("delete_segment", {"deleted_samples": (~mask).sum()}, start, end))
        self.dataChanged.emit()
        self.annotationsChanged.emit()
        self.historyChanged.emit()
        self.statusMessage.emit(f"Deleted {start:.3f}-{end:.3f} s")

    def mark_bad(self, start: float, end: float) -> None:
        if self.df is None or start >= end:
            self.statusMessage.emit("Invalid mask range")
            return
        self._push_state()
        mask = (self.df["normalized_time"] >= start) & (self.df["normalized_time"] <= end)
        self.df.loc[mask, "is_bad_segment"] = True
        self.history.append(OperationRecord("mark_bad", {}, start, end))
        self.dataChanged.emit()
        self.historyChanged.emit()
        self.statusMessage.emit(f"Marked bad {start:.3f}-{end:.3f} s")

    def annotate(self, start: float, end: float, label: str, track: str = "default", color: str = "#4e79a7") -> None:
        if self.df is None or start >= end:
            self.statusMessage.emit("Invalid annotation range")
            return
        self._push_state()
        ann = AnnotationSegment(start=start, end=end, label=label, track=track, color=color, id=self._id_counter)
        self._id_counter += 1
        self.annotations.append(ann)
        self.history.append(OperationRecord("annotate", {"label": label, "track": track}, start, end))
        self.annotationsChanged.emit()
        self.historyChanged.emit()
        self.statusMessage.emit(f"Annotated {start:.3f}-{end:.3f} s as {label}")

    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy() if self.df is not None else pd.DataFrame()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.df = df
        self._classify_columns(df)
        self._ensure_bad_mask()
        self.dataChanged.emit()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_clean(self, path: str) -> None:
        if self.df is None:
            self.statusMessage.emit("No data to save")
            return
        self.df.to_csv(path, index=False)
        self.statusMessage.emit(f"Saved cleaned CSV to {path}")

    def save_annotations(self, path: str) -> None:
        data = {
            "annotations": [ann.__dict__ for ann in self.annotations],
            "deletions": [{"start": s, "end": e} for s, e in self.deletions],
            "history": [record.__dict__ for record in self.history],
            "sample_rate": self.sample_rate,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.statusMessage.emit(f"Saved annotations to {path}")

    def load_annotations(self, path: str) -> None:
        if self.df is None:
            self.statusMessage.emit("Load data first")
            return
        if not os.path.isfile(path):
            self.statusMessage.emit(f"File not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        anns = data.get("annotations", [])
        dels = data.get("deletions", [])
        self.annotations = [AnnotationSegment(**a) for a in anns]
        self.deletions = [(d["start"], d["end"]) for d in dels]
        self.history = [OperationRecord(**h) for h in data.get("history", [])]
        if self.annotations:
            self._id_counter = max(a.id for a in self.annotations) + 1
        self.annotationsChanged.emit()
        self.historyChanged.emit()
        self.statusMessage.emit(f"Loaded annotations from {path}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def channel_groups(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {
            "Gaze": [],
            "Head": [],
            "Torso": [],
            "Feet": [],
            "Chair": [],
            "GMM": [],
            "Other": [],
        }
        for col in self.signal_columns:
            name = col.lower()
            if "gaze" in name:
                groups["Gaze"].append(col)
            elif "head" in name:
                groups["Head"].append(col)
            elif "torso" in name:
                groups["Torso"].append(col)
            elif "foot" in name:
                groups["Feet"].append(col)
            elif "chair" in name:
                groups["Chair"].append(col)
            elif "gmm" in name:
                groups["GMM"].append(col)
            else:
                groups["Other"].append(col)
        return groups

    def take_time_slice(self, start: float, end: float) -> pd.DataFrame:
        df = self.get_dataframe()
        if df.empty:
            return df
        return df[(df["normalized_time"] >= start) & (df["normalized_time"] <= end)].copy()

    def apply_dataframe(self, new_df: pd.DataFrame, description: str, start: float, end: float, params: Dict) -> None:
        self._push_state()
        self.df = new_df
        self._classify_columns(new_df)
        self._ensure_bad_mask()
        self.history.append(OperationRecord(description, params, start, end))
        self.dataChanged.emit()
        self.historyChanged.emit()

    def set_sample_rate(self, fs: float) -> None:
        self.sample_rate = float(fs)
        self.statusMessage.emit(f"Sampling rate set to {self.sample_rate} Hz")

