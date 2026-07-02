"""Invariant tests for diff-based undo/redo.

For every mutation kind: capture full state, mutate, undo (state must
equal the before-state exactly), redo (state must equal the after-state
exactly). Also pins the memory property that motivated the design:
metadata-only operations must not copy the DataFrame.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsdt_core.core import CoreDataModel


def make_model(rows: int = 200) -> CoreDataModel:
    m = CoreDataModel()
    df = pd.DataFrame({
        "normalized_time": np.arange(rows) / 100.0,
        "gaze_x": np.sin(np.arange(rows) / 10.0),
        "head_x": np.cos(np.arange(rows) / 10.0),
        "label_col": ["a"] * rows,
    })
    m.load_frame(df, "test.csv")
    return m


def snapshot(m: CoreDataModel) -> dict:
    return {
        "df": m.df.copy(),
        "annotations": [a.model_dump() for a in m.annotations],
        "deletions": list(m.deletions),
        "history": [h.model_dump() for h in m.history],
        "time_columns": list(m.time_columns),
        "metadata_columns": list(m.metadata_columns),
        "signal_columns": list(m.signal_columns),
    }


def assert_state(m: CoreDataModel, snap: dict) -> None:
    pd.testing.assert_frame_equal(m.df.reset_index(drop=True), snap["df"].reset_index(drop=True))
    assert [a.model_dump() for a in m.annotations] == snap["annotations"]
    assert list(m.deletions) == snap["deletions"]
    assert [h.model_dump() for h in m.history] == snap["history"]
    assert list(m.time_columns) == snap["time_columns"]
    assert list(m.metadata_columns) == snap["metadata_columns"]
    assert list(m.signal_columns) == snap["signal_columns"]


def roundtrip(m: CoreDataModel, mutate) -> None:
    """Assert mutate() is exactly reversed by undo and replayed by redo."""
    before = snapshot(m)
    mutate()
    after = snapshot(m)
    m.undo()
    assert_state(m, before)
    m.redo()
    assert_state(m, after)
    # And a second undo still works after the redo
    m.undo()
    assert_state(m, before)


# ---------------------------------------------------------------------------
# Per-operation invariants
# ---------------------------------------------------------------------------

def test_annotate_roundtrip():
    m = make_model()
    roundtrip(m, lambda: m.annotate(0.1, 0.3, "ev"))


def test_update_annotation_roundtrip():
    m = make_model()
    m.annotate(0.1, 0.3, "ev")
    ann_id = m.annotations[0].id
    roundtrip(m, lambda: m.update_annotation(ann_id, 0.15, 0.35, "changed", "t2", "#fff", 4))


def test_delete_annotation_roundtrip():
    m = make_model()
    m.annotate(0.1, 0.3, "ev")
    ann_id = m.annotations[0].id
    roundtrip(m, lambda: m.delete_annotation(ann_id))


def test_mark_bad_roundtrip():
    m = make_model()
    roundtrip(m, lambda: m.mark_bad(0.5, 0.8))


def test_delete_segment_roundtrip():
    m = make_model()
    roundtrip(m, lambda: m.delete_segment(0.5, 0.8))


def test_delete_segment_with_annotations_roundtrip():
    """Deletion adjusts annotations in place; undo must restore originals."""
    m = make_model()
    m.annotate(0.2, 0.4, "before")   # overlaps nothing
    m.annotate(0.6, 1.2, "spans")    # spans the deleted range end
    roundtrip(m, lambda: m.delete_segment(0.5, 0.8))


def test_delete_segment_preserve_gaps_roundtrip():
    m = make_model()
    m.preserve_timing_gaps = True
    roundtrip(m, lambda: m.delete_segment(0.5, 0.8))


def test_filter_apply_dataframe_roundtrip():
    m = make_model()
    filtered = m.get_dataframe()
    filtered["gaze_x"] = filtered["gaze_x"].rolling(5, min_periods=1).mean()
    roundtrip(m, lambda: m.apply_dataframe(
        filtered, "filter", 0.0, 2.0,
        {"channels": ["gaze_x"], "filter_type": "moving_average"},
    ))


def test_resample_apply_dataframe_roundtrip():
    """Length-changing frames must take the full-snapshot fallback."""
    m = make_model()
    resampled = m.get_dataframe().iloc[::2].reset_index(drop=True)
    roundtrip(m, lambda: m.apply_dataframe(
        resampled, "filter", 0.0, 2.0,
        {"channels": ["gaze_x"], "filter_type": "resample", "target_fs": 50.0},
    ))
    # roundtrip ends undone, so the entry (still full-snapshot kind) is on
    # the redo stack
    assert m._redo_stack[-1].kind == "full"


def test_rename_channels_roundtrip():
    m = make_model()
    roundtrip(m, lambda: m.rename_channels({"gaze_x": "eye_x", "head_x": "skull_x"}))


def test_delete_channels_roundtrip_preserves_position():
    m = make_model()
    before_order = list(m.df.columns)
    m.delete_channels(["gaze_x"])
    m.undo()
    assert list(m.df.columns) == before_order


def test_delete_channels_roundtrip():
    m = make_model()
    roundtrip(m, lambda: m.delete_channels(["gaze_x"]))


def test_duplicate_channels_roundtrip():
    m = make_model()
    roundtrip(m, lambda: m.duplicate_channels({"gaze_x": "gaze_x_copy"}))


def test_derived_channel_roundtrip():
    m = make_model()
    roundtrip(m, lambda: m.create_derived_channel("speed", "gaze_x + head_x"))


def test_derived_channel_overwrite_roundtrip():
    m = make_model()
    m.create_derived_channel("speed", "gaze_x + head_x")
    roundtrip(m, lambda: m.create_derived_channel("speed", "gaze_x * 2"))


# ---------------------------------------------------------------------------
# Sequences and memory properties
# ---------------------------------------------------------------------------

def test_mixed_sequence_full_unwind():
    m = make_model()
    states = [snapshot(m)]
    m.annotate(0.1, 0.3, "a"); states.append(snapshot(m))
    m.mark_bad(0.4, 0.6); states.append(snapshot(m))
    m.delete_segment(1.0, 1.2); states.append(snapshot(m))
    m.rename_channels({"gaze_x": "eye_x"}); states.append(snapshot(m))
    m.delete_channels(["head_x"]); states.append(snapshot(m))

    for expected in reversed(states[:-1]):
        m.undo()
        assert_state(m, expected)
    for expected in states[1:]:
        m.redo()
        assert_state(m, expected)


def test_metadata_ops_do_not_copy_dataframe():
    """The design goal: annotating must not snapshot the whole frame."""
    m = make_model(rows=100_000)
    df_mb = m.df.memory_usage(deep=True).sum() / (1024 * 1024)
    m.annotate(1.0, 2.0, "big")
    m.mark_bad(3.0, 4.0)
    stack_mb = m._estimate_undo_memory_mb()
    # mark_bad stores one boolean column; annotate stores no frame data.
    # Both together must be far below one frame copy.
    assert stack_mb < df_mb / 10, f"undo stack {stack_mb:.1f} MB vs frame {df_mb:.1f} MB"


def test_undo_kinds_are_diffs_not_snapshots():
    m = make_model()
    m.annotate(0.1, 0.2, "x")
    m.mark_bad(0.3, 0.4)
    m.delete_segment(0.5, 0.6)
    m.rename_channels({"gaze_x": "eye_x"})
    kinds = [e.kind for e in m._undo_stack]
    assert kinds == ["meta", "columns", "rows_insert", "rename"]


def test_undo_empty_stacks_are_safe():
    m = make_model()
    m.undo()  # must not raise
    m.redo()  # must not raise
    assert m.df is not None


def test_new_operation_clears_redo():
    m = make_model()
    m.annotate(0.1, 0.2, "a")
    m.undo()
    assert m._redo_stack
    m.mark_bad(0.3, 0.4)
    assert not m._redo_stack
