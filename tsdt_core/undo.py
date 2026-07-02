"""Diff-based undo entries for CoreDataModel.

Each UndoEntry stores the inverse payload of one mutation — only what is
needed to restore the previous state:

- "meta":        annotation/deletion/history changes only; no frame data
- "columns":     previous values of changed columns, names of added
                 columns (inverse drops them), and removed columns with
                 their positions (inverse reinserts them)
- "rows_insert": a deleted contiguous row block plus the previous time
                 axis (inverse reinserts the block)
- "rows_delete": the redo counterpart: drop a block and restore the
                 post-deletion time axis
- "rename":      a column-name mapping to apply in reverse
- "full":        complete DataFrame snapshot (fallback for anything that
                 cannot be expressed as a diff, e.g. resampling)

Every entry also carries the annotation list, deletion list, operation
history, and column-tracking lists — these are tiny next to the frame.
Applying an entry mutates the model back to the stored state and returns
the inverse entry, which makes undo/redo symmetric.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdt_core.core import CoreDataModel

# Rough per-object estimates for the metadata carried by every entry.
_ANNOTATION_BYTES = 200
_DELETION_BYTES = 32
_HISTORY_BYTES = 300


class UndoEntry:
    """The inverse of one mutation, applied to restore the prior state."""

    __slots__ = ("kind", "payload", "annotations", "deletions", "history", "tracking")

    def __init__(self, kind: str, payload: dict[str, Any], model: CoreDataModel) -> None:
        self.kind = kind
        self.payload = payload
        # Annotations are mutated in place elsewhere (update_annotation,
        # deletion adjustment), so they must be deep-copied here.
        self.annotations = [a.model_copy() for a in model.annotations]
        self.deletions = list(model.deletions)
        self.history = list(model.history)
        self.tracking = (
            list(model.time_columns),
            list(model.metadata_columns),
            list(model.signal_columns),
        )

    # ------------------------------------------------------------------
    def nbytes(self) -> int:
        total = (
            len(self.annotations) * _ANNOTATION_BYTES
            + len(self.deletions) * _DELETION_BYTES
            + len(self.history) * _HISTORY_BYTES
        )
        p = self.payload
        if self.kind == "full" and p.get("df") is not None:
            total += int(p["df"].memory_usage(deep=True).sum())
        elif self.kind == "columns":
            for series in p.get("changed", {}).values():
                total += int(series.memory_usage(deep=True))
            for _pos, series in p.get("removed", {}).values():
                total += int(series.memory_usage(deep=True))
        elif self.kind == "rows_insert":
            total += int(p["slice"].memory_usage(deep=True).sum())
            total += int(p["time"].memory_usage(deep=True))
        elif self.kind == "rows_delete":
            total += int(p["time"].memory_usage(deep=True))
        return total

    # ------------------------------------------------------------------
    def apply(self, model: CoreDataModel) -> UndoEntry:
        """Restore the stored state on model; return the inverse entry."""
        applier = getattr(self, f"_apply_{self.kind}")
        inverse = applier(model)
        # Metadata restore is common to all kinds. Copy again so the model
        # never shares mutable objects with stack entries.
        model.annotations = [a.model_copy() for a in self.annotations]
        model.deletions = list(self.deletions)
        model.history = list(self.history)
        model.time_columns = list(self.tracking[0])
        model.metadata_columns = list(self.tracking[1])
        model.signal_columns = list(self.tracking[2])
        return inverse

    # -- kind-specific appliers (each captures its inverse BEFORE mutating)

    def _apply_meta(self, model: CoreDataModel) -> UndoEntry:
        return UndoEntry("meta", {}, model)

    def _apply_full(self, model: CoreDataModel) -> UndoEntry:
        inverse = UndoEntry(
            "full",
            {"df": model.df.copy() if model.df is not None else None},
            model,
        )
        model.df = self.payload["df"]
        return inverse

    def _apply_rename(self, model: CoreDataModel) -> UndoEntry:
        mapping = self.payload["mapping"]
        inverse = UndoEntry(
            "rename", {"mapping": {v: k for k, v in mapping.items()}}, model
        )
        if model.df is not None:
            model.df.rename(columns=mapping, inplace=True)
        return inverse

    def _apply_columns(self, model: CoreDataModel) -> UndoEntry:
        df = model.df
        changed: dict[str, pd.Series] = self.payload.get("changed", {})
        added: list[str] = self.payload.get("added", [])
        removed: dict[str, tuple[int, pd.Series]] = self.payload.get("removed", {})

        inverse = UndoEntry(
            "columns",
            {
                "changed": {c: df[c].copy() for c in changed if c in df.columns},
                # Columns this entry reinserts must be dropped again on redo
                "added": list(removed.keys()),
                # Columns this entry drops must be reinserted on redo
                "removed": {
                    c: (int(df.columns.get_loc(c)), df[c].copy())
                    for c in added
                    if c in df.columns
                },
            },
            model,
        )

        for col in added:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        for col, series in changed.items():
            df[col] = series.copy()
        for col, (pos, series) in removed.items():
            df.insert(min(pos, len(df.columns)), col, series.copy())
        return inverse

    def _apply_rows_insert(self, model: CoreDataModel) -> UndoEntry:
        cur = model.df
        block: pd.DataFrame = self.payload["slice"]
        pos: int = self.payload["position"]
        inverse = UndoEntry(
            "rows_delete",
            {
                "position": pos,
                "count": len(block),
                "time": cur["normalized_time"].copy(),
            },
            model,
        )
        restored = pd.concat(
            [cur.iloc[:pos], block, cur.iloc[pos:]], ignore_index=True
        )
        restored["normalized_time"] = self.payload["time"].to_numpy()
        model.df = restored
        return inverse

    def _apply_rows_delete(self, model: CoreDataModel) -> UndoEntry:
        cur = model.df
        pos: int = self.payload["position"]
        count: int = self.payload["count"]
        inverse = UndoEntry(
            "rows_insert",
            {
                "slice": cur.iloc[pos:pos + count].copy(),
                "position": pos,
                "time": cur["normalized_time"].copy(),
            },
            model,
        )
        remaining = cur.drop(cur.index[pos:pos + count]).reset_index(drop=True)
        remaining["normalized_time"] = self.payload["time"].to_numpy()
        model.df = remaining
        return inverse
