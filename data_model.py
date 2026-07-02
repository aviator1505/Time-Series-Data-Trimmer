"""Core data model for time-series cleaning and annotation.

This module exposes DataModel which wraps a pandas DataFrame and
provides undo/redo, deletion with time collapse, masked segments,
annotation persistence, and operation history. It is UI-agnostic so it
can be reused in tests or other front-ends.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PySide6 import QtCore


@dataclass
class AnnotationSegment:
    start: float
    end: float
    label: str
    track: str = "default"
    color: str = "#4e79a7"
    id: int = field(default_factory=lambda: uuid.uuid4().int & 0x7FFFFFFF)
    episode_index: int | None = None  # Manual episode index override for CSV export


@dataclass
class OperationRecord:
    description: str
    params: dict
    start: float
    end: float


class DataModel(QtCore.QObject):
    """Backend for time-series data with undo/redo and annotation support."""

    # Epsilon for floating-point time comparisons (sub-nanosecond precision)
    TIME_EPSILON = 1e-9

    dataChanged = QtCore.Signal()
    annotationsChanged = QtCore.Signal()
    statusMessage = QtCore.Signal(str)
    historyChanged = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.df: pd.DataFrame | None = None
        self.original_df: pd.DataFrame | None = None
        self.time_columns: list[str] = []
        self.metadata_columns: list[str] = []
        self.signal_columns: list[str] = []
        self.annotations: list[AnnotationSegment] = []
        self.deletions: list[tuple[float, float]] = []
        self.history: list[OperationRecord] = []
        self.sample_rate: float = 120.0
        self._undo_stack: list[tuple[pd.DataFrame, list[AnnotationSegment], list[tuple[float, float]], list[OperationRecord]]] = []
        self._redo_stack: list[tuple[pd.DataFrame, list[AnnotationSegment], list[tuple[float, float]], list[OperationRecord]]] = []
        self._id_counter: int = 1
        # User preference: whether to preserve timing gaps after deletion
        self.preserve_timing_gaps: bool = False

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
        metadata_cols: list[str] = []
        signal_cols: list[str] = []
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
    MAX_UNDO_STATES = 30  # Limit number of undo states (fallback limit)
    MAX_UNDO_MEMORY_MB = 500  # Maximum memory for undo stack in MB

    def _estimate_undo_memory_mb(self) -> float:
        """Estimate memory usage of undo stack in MB.

        Calculates the approximate memory footprint of all DataFrame copies
        stored in the undo stack. This is used to enforce memory limits and
        provide user feedback about memory consumption.

        Returns:
            Estimated memory usage in megabytes.
        """
        total_bytes = 0
        for state in self._undo_stack:
            df, annotations, deletions, history = state
            # DataFrame memory (deep=True includes object dtype overhead)
            total_bytes += df.memory_usage(deep=True).sum()
            # Annotations, deletions, and history are small relative to DataFrames,
            # but we add a rough estimate for completeness
            total_bytes += len(annotations) * 200  # ~200 bytes per annotation object
            total_bytes += len(deletions) * 32  # ~32 bytes per tuple (two floats)
            total_bytes += len(history) * 300  # ~300 bytes per OperationRecord
        return total_bytes / (1024 * 1024)

    def _push_state(self) -> None:
        if self.df is None:
            return
        self._undo_stack.append(
            (self.df.copy(), list(self.annotations), list(self.deletions), list(self.history))
        )
        self._redo_stack.clear()

        # Prune oldest states if memory exceeds limit
        pruned_for_memory = False
        while self._estimate_undo_memory_mb() > self.MAX_UNDO_MEMORY_MB and len(self._undo_stack) > 1:
            self._undo_stack.pop(0)
            pruned_for_memory = True
        if pruned_for_memory:
            mem_mb = self._estimate_undo_memory_mb()
            self.statusMessage.emit(f"Undo stack pruned due to memory limit ({mem_mb:.1f} MB)")

        # Fallback: prune oldest states if stack exceeds count limit
        while len(self._undo_stack) > self.MAX_UNDO_STATES:
            self._undo_stack.pop(0)

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
    def _adjust_annotations_after_deletion(self, start: float, end: float) -> None:
        """Adjust annotation boundaries after a segment deletion.

        This method modifies self.annotations in-place to account for the
        collapsed timeline after a segment [start, end] is deleted.

        Rules applied (in order of precedence):
        - Annotation entirely BEFORE deletion (ann.end <= start): Keep unchanged
        - Annotation entirely AFTER deletion (ann.start >= end): Shift backwards by deletion_duration
        - Annotation entirely INSIDE deletion (ann.start >= start AND ann.end <= end): Remove
        - Annotation SPANS deletion (ann.start < start AND ann.end > end): Shrink end by deletion_duration
        - Annotation overlaps START of deletion (ann.start < start AND ann.end <= end): Truncate ann.end to start
        - Annotation overlaps END of deletion (ann.start >= start AND ann.end > end): Set ann.start = start, shift ann.end

        Args:
            start: Start time of the deleted segment
            end: End time of the deleted segment
        """
        if not self.annotations:
            return

        deletion_duration = end - start
        adjusted_annotations: list[AnnotationSegment] = []

        for ann in self.annotations:
            # Case 1: Entirely BEFORE deletion - keep unchanged
            if ann.end <= start:
                adjusted_annotations.append(ann)
                continue

            # Case 2: Entirely AFTER deletion - shift backwards
            if ann.start >= end:
                ann.start -= deletion_duration
                ann.end -= deletion_duration
                adjusted_annotations.append(ann)
                continue

            # Case 3: Entirely INSIDE deletion - remove (skip appending)
            if ann.start >= start and ann.end <= end:
                # Annotation is completely within the deleted region - remove it
                continue

            # Case 4: SPANS deletion (starts before, ends after)
            if ann.start < start and ann.end > end:
                # Shrink the annotation by the deletion duration
                ann.end -= deletion_duration
                adjusted_annotations.append(ann)
                continue

            # Case 5: Overlaps START of deletion (starts before, ends inside/at deletion)
            if ann.start < start and ann.end > start and ann.end <= end:
                # Truncate the end to the deletion start
                ann.end = start
                # Only keep if annotation still has positive duration
                if ann.end > ann.start:
                    adjusted_annotations.append(ann)
                continue

            # Case 6: Overlaps END of deletion (starts inside/at deletion, ends after)
            if ann.start >= start and ann.start < end and ann.end > end:
                # The portion after deletion becomes the annotation
                # New start is at the deletion start (where the post-deletion content now begins)
                ann.start = start
                # Shift end backwards by deletion duration
                ann.end -= deletion_duration
                # Only keep if annotation still has positive duration
                if ann.end > ann.start:
                    adjusted_annotations.append(ann)
                continue

        self.annotations = adjusted_annotations

    def delete_segment(self, start: float, end: float) -> None:
        if self.df is None or start >= end:
            self.statusMessage.emit("Invalid delete range")
            return
        self._push_state()

        # Use epsilon-tolerant comparison to handle floating-point precision
        time_col = self.df["normalized_time"].values
        in_segment = (time_col >= start - self.TIME_EPSILON) & (time_col <= end + self.TIME_EPSILON)
        mask = ~in_segment
        deleted_count = in_segment.sum()

        new_df = self.df.loc[mask].copy().reset_index(drop=True)

        if self.preserve_timing_gaps:
            # Option A: Keep original timestamps (preserves gaps in timeline)
            # This is useful for analyzing timing patterns in scientific data
            pass
        else:
            # Option B: Shift subsequent timestamps to close the gap (default)
            # This maintains a continuous timeline after deletion
            deletion_duration = end - start
            if "normalized_time" in new_df.columns and len(new_df) > 0:
                new_time = new_df["normalized_time"].values.copy()
                # Shift all timestamps that were after the deletion
                post_deletion_mask = new_time > (end + self.TIME_EPSILON)
                new_time[post_deletion_mask] -= deletion_duration
                # Round to millisecond precision for cleaner display
                new_df["normalized_time"] = np.round(new_time, 3)

        self.df = new_df
        self.deletions.append((start, end))
        self.history.append(OperationRecord("delete_segment", {"deleted_samples": int(deleted_count)}, start, end))
        # Adjust annotation boundaries to account for collapsed timeline
        self._adjust_annotations_after_deletion(start, end)
        self.dataChanged.emit()
        self.annotationsChanged.emit()
        self.historyChanged.emit()
        self.statusMessage.emit(f"Deleted {start:.3f}-{end:.3f} s ({deleted_count} samples)")

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

    def annotate(self, start: float, end: float, label: str, track: str = "episode", color: str = "#4e79a7") -> None:
        if self.df is None or start >= end:
            self.statusMessage.emit("Invalid annotation range")
            return
        self._push_state()
        # Ensure unique ID by finding max existing ID
        if self.annotations:
            max_existing_id = max(a.id for a in self.annotations)
            self._id_counter = max(self._id_counter, max_existing_id + 1)
        ann = AnnotationSegment(start=start, end=end, label=label, track=track, color=color, id=self._id_counter)
        self._id_counter += 1
        self.annotations.append(ann)
        self.history.append(OperationRecord("annotate", {"label": label, "track": track}, start, end))
        self.annotationsChanged.emit()
        self.historyChanged.emit()
        self.statusMessage.emit(f"Annotated {start:.3f}-{end:.3f} s as {label}")

    def update_annotation(
        self,
        ann_id: int,
        start: float,
        end: float,
        label: str | None,
        track: str | None,
        color: str | None,
        episode_index: int | None = None
    ) -> None:
        self._push_state()  # Capture state before mutation for undo support
        for ann in self.annotations:
            if ann.id == ann_id:
                ann.start = start
                ann.end = end
                if label is not None:
                    ann.label = label
                if track is not None:
                    ann.track = track
                if color is not None:
                    ann.color = color
                # episode_index can be set to None (auto) or a specific value
                ann.episode_index = episode_index
                self.annotationsChanged.emit()
                self.historyChanged.emit()
                self.statusMessage.emit(f"Updated annotation {ann_id}")
                break

    def delete_annotation(self, ann_id: int) -> None:
        self._push_state()  # Capture state before mutation for undo support
        self.annotations = [a for a in self.annotations if a.id != ann_id]
        self.annotationsChanged.emit()
        self.historyChanged.emit()

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
    def save_clean(
        self,
        path: str,
        embed_annotations: bool = True,
        manual_indices: dict[int, int] | None = None
    ) -> None:
        """Save cleaned DataFrame to CSV, optionally embedding annotations.

        Args:
            path: Output file path
            embed_annotations: If True, write annotations as episode columns
            manual_indices: Optional dict mapping annotation.id -> desired episode_index
        """
        if self.df is None:
            self.statusMessage.emit("No data to save")
            return

        output_df = self.df.copy()

        if embed_annotations and self.annotations:
            output_df = self.annotations_to_episode_columns(
                output_df,
                self.annotations,
                manual_indices
            )

        output_df.to_csv(path, index=False)
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
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        anns = data.get("annotations", [])
        dels = data.get("deletions", [])

        # Robust annotation deserialization: skip malformed entries
        parsed_annotations: list[AnnotationSegment] = []
        skipped_count = 0
        for a in anns:
            try:
                parsed_annotations.append(AnnotationSegment(**a))
            except (TypeError, ValueError):
                skipped_count += 1
                continue
        self.annotations = parsed_annotations
        if skipped_count > 0:
            self.statusMessage.emit(f"Skipped {skipped_count} malformed annotations")
        parsed_deletions: list[tuple[float, float]] = []
        for d in dels:
            if isinstance(d, dict) and "start" in d and "end" in d:
                try:
                    parsed_deletions.append((float(d["start"]), float(d["end"])))
                except (TypeError, ValueError):
                    continue
            elif isinstance(d, (list, tuple)) and len(d) == 2:
                try:
                    parsed_deletions.append((float(d[0]), float(d[1])))
                except (TypeError, ValueError):
                    continue
        self.deletions = parsed_deletions
        self.history = [OperationRecord(**h) for h in data.get("history", [])]
        if "sample_rate" in data:
            try:
                self.sample_rate = float(data["sample_rate"])
            except (TypeError, ValueError):
                pass
        if self.annotations:
            self._id_counter = max(a.id for a in self.annotations) + 1
        self.annotationsChanged.emit()
        self.historyChanged.emit()
        self.statusMessage.emit(f"Loaded annotations from {path}")

    # ------------------------------------------------------------------
    # Episode column conversion (annotations <-> CSV columns)
    # ------------------------------------------------------------------
    def _parse_annotation_label(self, label: str) -> tuple[str, str]:
        """Parse annotation label into (episode_type, episode_state).

        Handles formats:
        - "episode:inspection:inspecting_screen" -> ("inspection", "inspecting_screen")
        - "episode:action" -> ("action", "")
        - "blink" -> ("blink", "")  # Non-standard labels become type
        """
        if label.startswith("episode:"):
            parts = label.split(":", 2)  # Split into at most 3 parts
            if len(parts) >= 3:
                return (parts[1], parts[2])
            elif len(parts) == 2:
                return (parts[1], "")
        # Non-episode labels: use label as type, empty state
        return (label, "")

    def _assign_episode_indices(
        self,
        annotations: list[AnnotationSegment],
        manual_indices: dict[int, int] | None = None
    ) -> dict[int, int]:
        """Assign episode indices to annotations.

        Priority:
        1. annotation.episode_index if set on the object
        2. manual_indices dict if provided
        3. Auto-assign sequential indices by start time

        Args:
            annotations: List of annotations to assign indices
            manual_indices: Optional dict mapping annotation.id -> desired episode_index

        Returns:
            Dict mapping annotation.id -> assigned episode_index
        """
        if not annotations:
            return {}

        # Sort by start time (then by end time for ties)
        sorted_anns = sorted(annotations, key=lambda a: (a.start, a.end))

        # Build combined manual indices from annotation.episode_index and manual_indices param
        combined_manual: dict[int, int] = {}
        for ann in sorted_anns:
            if ann.episode_index is not None:
                combined_manual[ann.id] = ann.episode_index
        if manual_indices:
            combined_manual.update(manual_indices)  # Param overrides annotation field

        if not combined_manual:
            # Auto-assign: sequential indices starting from 1
            return {ann.id: idx + 1 for idx, ann in enumerate(sorted_anns)}

        # Manual assignment with shifting
        result: dict[int, int] = {}
        used_indices: set = set(combined_manual.values())

        # First pass: assign manual indices
        for ann_id, desired_idx in combined_manual.items():
            result[ann_id] = desired_idx

        # Second pass: assign remaining indices, shifting as needed
        next_available = 1
        for ann in sorted_anns:
            if ann.id in result:
                continue
            # Find next available index that doesn't conflict
            while next_available in used_indices:
                next_available += 1
            result[ann.id] = next_available
            used_indices.add(next_available)
            next_available += 1

        return result

    def annotations_to_episode_columns(
        self,
        df: pd.DataFrame,
        annotations: list[AnnotationSegment],
        manual_indices: dict[int, int] | None = None
    ) -> pd.DataFrame:
        """Embed annotations as episode columns in DataFrame.

        Creates/overwrites columns:
        - episode_index: Integer identifier for each episode
        - episode_type: String label (e.g., "inspection", "action")
        - episode_state: Optional state descriptor

        Args:
            df: DataFrame to modify (will be copied)
            annotations: All annotations to embed
            manual_indices: Optional dict mapping annotation.id -> desired index

        Returns:
            Modified DataFrame with episode columns
        """
        if df.empty or not annotations:
            return df.copy()

        result_df = df.copy()
        time_col = result_df["normalized_time"].values

        # Initialize columns with NaN/empty
        result_df["episode_index"] = np.nan
        result_df["episode_type"] = ""
        result_df["episode_state"] = ""

        # Assign indices
        index_map = self._assign_episode_indices(annotations, manual_indices)

        # Sort annotations by start time for consistent processing
        sorted_anns = sorted(annotations, key=lambda a: a.start)

        for ann in sorted_anns:
            # Find rows within annotation time range
            mask = (time_col >= ann.start) & (time_col <= ann.end)

            if not mask.any():
                continue

            episode_idx = index_map[ann.id]
            episode_type, episode_state = self._parse_annotation_label(ann.label)

            # Assign to matching rows
            result_df.loc[mask, "episode_index"] = episode_idx
            result_df.loc[mask, "episode_type"] = episode_type
            result_df.loc[mask, "episode_state"] = episode_state

        return result_df

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def channel_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            "Time / LSL": [],
            "Gaze": [],
            "Head": [],
            "Chest/Torso": [],
            "Feet": [],
            "Chair": [],
            "Workspace": [],
            "Screen": [],
            "Position": [],
            "Orientation/Quat": [],
            "Fixation": [],
            "Other": [],
        }

        def assign(col: str, name: str) -> None:
            # helper to append to group map
            groups[name].append(col)

        for col in self.signal_columns:
            name = col.lower().replace(" ", "_")
            if any(tok in name for tok in ["normalized_time", "timestamp", "lsl"]):
                assign(col, "Time / LSL")
            elif "gaze" in name:
                assign(col, "Gaze")
            elif "head" in name or "mocap_head" in name:
                assign(col, "Head")
            elif "chest" in name or "torso" in name:
                assign(col, "Chest/Torso")
            elif "foot" in name:
                assign(col, "Feet")
            elif "chair" in name:
                assign(col, "Chair")
            elif "workspace" in name:
                assign(col, "Workspace")
            elif "screen" in name:
                assign(col, "Screen")
            elif "fix" in name:
                assign(col, "Fixation")
            elif name.endswith(("_x", "_y", "_z")) or any(s in name for s in ["_x_", "_y_", "_z_"]):
                assign(col, "Position")
            elif any(tok in name for tok in ["quat", "qx", "qy", "qz", "qw", "azimuth", "elevation", "angle"]):
                assign(col, "Orientation/Quat")
            else:
                assign(col, "Other")
        # prune empty groups for cleaner UI
        return {k: v for k, v in groups.items() if v}

    def take_time_slice(self, start: float, end: float) -> pd.DataFrame:
        df = self.get_dataframe()
        if df.empty:
            return df
        return df[(df["normalized_time"] >= start) & (df["normalized_time"] <= end)].copy()

    def apply_dataframe(self, new_df: pd.DataFrame, description: str, start: float, end: float, params: dict) -> None:
        self._push_state()
        self.df = new_df
        self._classify_columns(new_df)
        self._ensure_bad_mask()
        self.history.append(OperationRecord(description, params, start, end))
        self.dataChanged.emit()
        self.historyChanged.emit()

    def rename_channels(self, mappings: dict[str, str]) -> None:
        """Rename columns in DataFrame and update tracking lists.

        Args:
            mappings: Dict of {old_name: new_name}
        """
        if not mappings or self.df is None:
            return

        self._push_state()

        # Apply rename to DataFrame
        self.df.rename(columns=mappings, inplace=True)

        # Update column tracking lists (preserve order)
        for old_col, new_col in mappings.items():
            if old_col in self.time_columns:
                idx = self.time_columns.index(old_col)
                self.time_columns[idx] = new_col
            elif old_col in self.metadata_columns:
                idx = self.metadata_columns.index(old_col)
                self.metadata_columns[idx] = new_col
            elif old_col in self.signal_columns:
                idx = self.signal_columns.index(old_col)
                self.signal_columns[idx] = new_col

        # Record in history for recipe generation
        end_time = self.df["normalized_time"].max() if "normalized_time" in self.df.columns else 0.0
        self.history.append(OperationRecord(
            "rename",
            {"mappings": mappings},
            0.0,
            end_time
        ))

        self.dataChanged.emit()
        self.historyChanged.emit()

    def delete_channels(self, columns: list[str]) -> None:
        """Delete columns from DataFrame and tracking lists.

        Args:
            columns: List of column names to delete
        """
        if not columns or self.df is None:
            return

        self._push_state()

        # Remove columns from DataFrame (only those that exist)
        cols_to_drop = [c for c in columns if c in self.df.columns]
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)

        # Update tracking lists
        self.signal_columns = [c for c in self.signal_columns if c not in columns]
        self.metadata_columns = [c for c in self.metadata_columns if c not in columns]
        self.time_columns = [c for c in self.time_columns if c not in columns]

        # Record in history
        end_time = self.df["normalized_time"].max() if "normalized_time" in self.df.columns else 0.0
        self.history.append(OperationRecord(
            "delete_channels",
            {"columns": columns},
            0.0,
            end_time
        ))

        self.dataChanged.emit()
        self.historyChanged.emit()

    def duplicate_channels(self, mappings: dict[str, str]) -> None:
        """Duplicate columns with new names.

        Args:
            mappings: Dict of {source_col: new_col_name}
        """
        if not mappings or self.df is None:
            return

        self._push_state()

        # Copy each column
        for source, new_name in mappings.items():
            if source in self.df.columns:
                self.df[new_name] = self.df[source].copy()
                # Add to appropriate tracking list
                if source in self.signal_columns:
                    self.signal_columns.append(new_name)
                elif source in self.metadata_columns:
                    self.metadata_columns.append(new_name)

        # Record in history
        end_time = self.df["normalized_time"].max() if "normalized_time" in self.df.columns else 0.0
        self.history.append(OperationRecord(
            "duplicate_channels",
            {"mappings": mappings},
            0.0,
            end_time
        ))

        self.dataChanged.emit()
        self.historyChanged.emit()

    def create_derived_channel(self, name: str, expr: str) -> None:
        """Create a new channel from an expression.

        Args:
            name: Name for the new channel
            expr: pd.eval() compatible expression referencing existing columns
        """
        if not name or not expr or self.df is None:
            return

        self._push_state()

        # Evaluate expression
        self.df[name] = pd.eval(expr, local_dict=self.df.to_dict("series"))

        # Add to signal columns (derived channels are always numeric)
        if name not in self.signal_columns:
            self.signal_columns.append(name)

        # Record in history
        end_time = self.df["normalized_time"].max() if "normalized_time" in self.df.columns else 0.0
        self.history.append(OperationRecord(
            "derived",
            {"name": name, "expr": expr},
            0.0,
            end_time
        ))

        self.dataChanged.emit()
        self.historyChanged.emit()

    def set_sample_rate(self, fs: float, recalculate_time: bool = False) -> None:
        """Set the sample rate.

        Args:
            fs: New sample rate in Hz
            recalculate_time: If True, regenerate normalized_time based on new rate
        """
        old_rate = self.sample_rate
        self.sample_rate = float(fs)

        if recalculate_time and self.df is not None and "normalized_time" in self.df.columns:
            # Regenerate time axis based on new sample rate
            n = len(self.df)
            self.df["normalized_time"] = np.arange(n) / self.sample_rate
            self.dataChanged.emit()
            self.statusMessage.emit(f"Sampling rate changed from {old_rate:.1f} to {self.sample_rate:.1f} Hz (time axis recalculated)")
        else:
            self.statusMessage.emit(f"Sampling rate set to {self.sample_rate} Hz")
