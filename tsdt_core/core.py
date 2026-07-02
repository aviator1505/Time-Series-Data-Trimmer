"""Headless core data model for time-series cleaning and annotation.

CoreDataModel wraps a pandas DataFrame and provides undo/redo, deletion
with time collapse, masked segments, annotation persistence, and
operation history. It has no Qt dependency: front-ends observe it by
overriding the _notify_* hooks (see data_model.DataModel for the Qt
adapter that turns them into signals).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from tsdt_core.models import AnnotationSegment, OperationRecord
from tsdt_core.undo import UndoEntry


class CoreDataModel:
    """Backend for time-series data with undo/redo and annotation support."""

    # Epsilon for floating-point time comparisons (sub-nanosecond precision)
    TIME_EPSILON = 1e-9



    # ------------------------------------------------------------------
    # Observer hooks (no-ops here; the Qt adapter overrides these to emit
    # signals, other front-ends may override them however they need)
    # ------------------------------------------------------------------
    def _notify_data_changed(self) -> None:
        pass

    def _notify_annotations_changed(self) -> None:
        pass

    def _notify_history_changed(self) -> None:
        pass

    def _notify_status(self, message: str) -> None:
        pass

    def __init__(self) -> None:
        self.df: pd.DataFrame | None = None
        self.original_df: pd.DataFrame | None = None
        self.time_columns: list[str] = []
        self.metadata_columns: list[str] = []
        self.signal_columns: list[str] = []
        self.annotations: list[AnnotationSegment] = []
        self.deletions: list[tuple[float, float]] = []
        self.history: list[OperationRecord] = []
        self.sample_rate: float = 120.0
        self._undo_stack: list[UndoEntry] = []
        self._redo_stack: list[UndoEntry] = []
        self._id_counter: int = 1
        # User preference: whether to preserve timing gaps after deletion
        self.preserve_timing_gaps: bool = False
        # Report from the last adaptive ingest (None until a file is loaded)
        self.ingest_report = None

    # ------------------------------------------------------------------
    # Loading and classification
    # ------------------------------------------------------------------
    def load_csv(self, path: str) -> None:
        df = self.read_csv_frame(path)
        self.load_frame(df, path)

    @staticmethod
    def read_csv_frame(path: str) -> pd.DataFrame:
        """Parse a tabular file into a normalized DataFrame.

        Uses adaptive ingestion (delimiter/encoding sniffing, numeric
        coercion, time-unit detection); the IngestReport rides along in
        df.attrs["ingest_report"]. Pure compute with no model mutation,
        so it is safe to run on a worker thread; pass the result to
        load_frame on the UI thread.
        """
        from tsdt_core.ingest import smart_read

        df, report = smart_read(path)
        df.attrs["ingest_report"] = report
        return df

    def load_frame(self, df: pd.DataFrame, path: str) -> None:
        """Adopt a parsed DataFrame as the new session (resets all state)."""
        self.ingest_report = df.attrs.get("ingest_report")
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
        self._notify_data_changed()
        message = f"Loaded {os.path.basename(path)}"
        detected = self.ingest_report.summary() if self.ingest_report is not None else ""
        if detected:
            message += f" | detected: {detected}"
        self._notify_status(message)

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
        """Estimate memory usage of the undo stack in MB.

        With diff-based entries this is the payload size of each entry
        (changed columns, deleted row blocks, or a full frame only for
        fallback snapshots), used to enforce the memory cap.
        """
        return sum(entry.nbytes() for entry in self._undo_stack) / (1024 * 1024)

    def _push_entry(self, entry: UndoEntry) -> None:
        self._undo_stack.append(entry)
        self._redo_stack.clear()

        # Prune oldest states if memory exceeds limit
        pruned_for_memory = False
        while self._estimate_undo_memory_mb() > self.MAX_UNDO_MEMORY_MB and len(self._undo_stack) > 1:
            self._undo_stack.pop(0)
            pruned_for_memory = True
        if pruned_for_memory:
            mem_mb = self._estimate_undo_memory_mb()
            self._notify_status(f"Undo stack pruned due to memory limit ({mem_mb:.1f} MB)")

        # Fallback: prune oldest states if stack exceeds count limit
        while len(self._undo_stack) > self.MAX_UNDO_STATES:
            self._undo_stack.pop(0)

    def _push_state(self) -> None:
        """Push a full snapshot (fallback for non-diffable mutations)."""
        if self.df is None:
            return
        self._push_entry(UndoEntry("full", {"df": self.df.copy()}, self))

    def _push_meta(self) -> None:
        """Push an entry for mutations that touch no frame data."""
        if self.df is None:
            return
        self._push_entry(UndoEntry("meta", {}, self))

    def _push_columns(
        self,
        changed: list[str] | None = None,
        added: list[str] | None = None,
        removed: list[str] | None = None,
    ) -> None:
        """Push a column-level diff: previous values of `changed` columns,
        names of columns about to be `added` (undo drops them), and data +
        positions of columns about to be `removed` (undo reinserts them)."""
        if self.df is None:
            return
        df = self.df
        payload = {
            "changed": {c: df[c].copy() for c in (changed or []) if c in df.columns},
            "added": list(added or []),
            "removed": {
                c: (int(df.columns.get_loc(c)), df[c].copy())
                for c in (removed or [])
                if c in df.columns
            },
        }
        self._push_entry(UndoEntry("columns", payload, self))

    def _push_rows_removed(self, block: pd.DataFrame, position: int) -> None:
        """Push a row-level diff for a contiguous block about to be deleted."""
        if self.df is None:
            return
        payload = {
            "slice": block,
            "position": int(position),
            "time": self.df["normalized_time"].copy(),
        }
        self._push_entry(UndoEntry("rows_insert", payload, self))

    def _push_rename(self, mapping: dict[str, str]) -> None:
        """Push a rename inverse ({new: old})."""
        if self.df is None:
            return
        self._push_entry(
            UndoEntry("rename", {"mapping": {v: k for k, v in mapping.items()}}, self)
        )

    def undo(self) -> None:
        if not self._undo_stack:
            self._notify_status("Nothing to undo")
            return
        entry = self._undo_stack.pop()
        self._redo_stack.append(entry.apply(self))
        self._notify_data_changed()
        self._notify_annotations_changed()
        self._notify_history_changed()
        self._notify_status("Undo")

    def redo(self) -> None:
        if not self._redo_stack:
            self._notify_status("Nothing to redo")
            return
        entry = self._redo_stack.pop()
        self._undo_stack.append(entry.apply(self))
        self._notify_data_changed()
        self._notify_annotations_changed()
        self._notify_history_changed()
        self._notify_status("Redo")

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
            self._notify_status("Invalid delete range")
            return

        # Use epsilon-tolerant comparison to handle floating-point precision
        time_col = self.df["normalized_time"].values
        in_segment = (time_col >= start - self.TIME_EPSILON) & (time_col <= end + self.TIME_EPSILON)
        mask = ~in_segment
        deleted_count = in_segment.sum()

        # With a monotonic time axis the deleted block is contiguous and a
        # cheap row-diff suffices; otherwise fall back to a full snapshot.
        positions = np.flatnonzero(in_segment)
        if len(positions) > 0 and np.all(np.diff(positions) == 1):
            self._push_rows_removed(
                self.df.iloc[positions[0]:positions[-1] + 1].copy(), positions[0]
            )
        else:
            self._push_state()

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
        self.history.append(OperationRecord(description="delete_segment", params={"deleted_samples": int(deleted_count)}, start=start, end=end))
        # Adjust annotation boundaries to account for collapsed timeline
        self._adjust_annotations_after_deletion(start, end)
        self._notify_data_changed()
        self._notify_annotations_changed()
        self._notify_history_changed()
        self._notify_status(f"Deleted {start:.3f}-{end:.3f} s ({deleted_count} samples)")

    def mark_bad(self, start: float, end: float) -> None:
        if self.df is None or start >= end:
            self._notify_status("Invalid mask range")
            return
        self._push_columns(changed=["is_bad_segment"])
        mask = (self.df["normalized_time"] >= start) & (self.df["normalized_time"] <= end)
        self.df.loc[mask, "is_bad_segment"] = True
        self.history.append(OperationRecord(description="mark_bad", params={}, start=start, end=end))
        self._notify_data_changed()
        self._notify_history_changed()
        self._notify_status(f"Marked bad {start:.3f}-{end:.3f} s")

    def annotate(self, start: float, end: float, label: str, track: str = "episode", color: str = "#4e79a7") -> None:
        if self.df is None or start >= end:
            self._notify_status("Invalid annotation range")
            return
        self._push_meta()
        # Ensure unique ID by finding max existing ID
        if self.annotations:
            max_existing_id = max(a.id for a in self.annotations)
            self._id_counter = max(self._id_counter, max_existing_id + 1)
        ann = AnnotationSegment(start=start, end=end, label=label, track=track, color=color, id=self._id_counter)
        self._id_counter += 1
        self.annotations.append(ann)
        self.history.append(OperationRecord(description="annotate", params={"label": label, "track": track}, start=start, end=end))
        self._notify_annotations_changed()
        self._notify_history_changed()
        self._notify_status(f"Annotated {start:.3f}-{end:.3f} s as {label}")

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
        self._push_meta()  # Capture state before mutation for undo support
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
                self._notify_annotations_changed()
                self._notify_history_changed()
                self._notify_status(f"Updated annotation {ann_id}")
                break

    def delete_annotation(self, ann_id: int) -> None:
        self._push_meta()  # Capture state before mutation for undo support
        self.annotations = [a for a in self.annotations if a.id != ann_id]
        self._notify_annotations_changed()
        self._notify_history_changed()

    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy() if self.df is not None else pd.DataFrame()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.df = df
        self._classify_columns(df)
        self._ensure_bad_mask()
        self._notify_data_changed()

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
            self._notify_status("No data to save")
            return

        output_df = self.df.copy()

        if embed_annotations and self.annotations:
            output_df = self.annotations_to_episode_columns(
                output_df,
                self.annotations,
                manual_indices
            )

        output_df.to_csv(path, index=False)
        self._notify_status(f"Saved cleaned CSV to {path}")

    def save_annotations(self, path: str) -> None:
        data = {
            "annotations": [ann.model_dump() for ann in self.annotations],
            "deletions": [{"start": s, "end": e} for s, e in self.deletions],
            "history": [record.model_dump() for record in self.history],
            "sample_rate": self.sample_rate,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._notify_status(f"Saved annotations to {path}")

    def load_annotations(self, path: str) -> None:
        if self.df is None:
            self._notify_status("Load data first")
            return
        if not os.path.isfile(path):
            self._notify_status(f"File not found: {path}")
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
            self._notify_status(f"Skipped {skipped_count} malformed annotations")
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
        self._notify_annotations_changed()
        self._notify_history_changed()
        self._notify_status(f"Loaded annotations from {path}")

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
        # Same-length frames with a declared channel list (the filter path)
        # only need a column diff; anything else (resample, unknown scope)
        # falls back to a full snapshot.
        channels = params.get("channels")
        old = self.df
        if (
            old is not None
            and channels
            and len(new_df) == len(old)
            and not (set(old.columns) - set(new_df.columns))
        ):
            added = [c for c in new_df.columns if c not in old.columns]
            self._push_columns(changed=list(channels), added=added)
        else:
            self._push_state()
        self.df = new_df
        self._classify_columns(new_df)
        self._ensure_bad_mask()
        self.history.append(OperationRecord(description=description, params=params, start=start, end=end))
        self._notify_data_changed()
        self._notify_history_changed()

    def rename_channels(self, mappings: dict[str, str]) -> None:
        """Rename columns in DataFrame and update tracking lists.

        Args:
            mappings: Dict of {old_name: new_name}
        """
        if not mappings or self.df is None:
            return

        self._push_rename(mappings)

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
            description="rename",
            params={"mappings": mappings},
            start=0.0,
            end=end_time
        ))

        self._notify_data_changed()
        self._notify_history_changed()

    def delete_channels(self, columns: list[str]) -> None:
        """Delete columns from DataFrame and tracking lists.

        Args:
            columns: List of column names to delete
        """
        if not columns or self.df is None:
            return

        self._push_columns(removed=columns)

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
            description="delete_channels",
            params={"columns": columns},
            start=0.0,
            end=end_time
        ))

        self._notify_data_changed()
        self._notify_history_changed()

    def duplicate_channels(self, mappings: dict[str, str]) -> None:
        """Duplicate columns with new names.

        Args:
            mappings: Dict of {source_col: new_col_name}
        """
        if not mappings or self.df is None:
            return

        self._push_columns(added=[n for n in mappings.values() if n not in self.df.columns])

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
            description="duplicate_channels",
            params={"mappings": mappings},
            start=0.0,
            end=end_time
        ))

        self._notify_data_changed()
        self._notify_history_changed()

    def create_derived_channel(self, name: str, expr: str) -> None:
        """Create a new channel from an expression.

        Args:
            name: Name for the new channel
            expr: pd.eval() compatible expression referencing existing columns
        """
        if not name or not expr or self.df is None:
            return

        # A derived channel may overwrite an existing column; store its
        # previous values in that case, otherwise just record the addition.
        if name in self.df.columns:
            self._push_columns(changed=[name])
        else:
            self._push_columns(added=[name])

        # Evaluate expression
        self.df[name] = pd.eval(expr, local_dict=self.df.to_dict("series"))

        # Add to signal columns (derived channels are always numeric)
        if name not in self.signal_columns:
            self.signal_columns.append(name)

        # Record in history
        end_time = self.df["normalized_time"].max() if "normalized_time" in self.df.columns else 0.0
        self.history.append(OperationRecord(
            description="derived",
            params={"name": name, "expr": expr},
            start=0.0,
            end=end_time
        ))

        self._notify_data_changed()
        self._notify_history_changed()

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
            self._notify_data_changed()
            self._notify_status(f"Sampling rate changed from {old_rate:.1f} to {self.sample_rate:.1f} Hz (time axis recalculated)")
        else:
            self._notify_status(f"Sampling rate set to {self.sample_rate} Hz")
