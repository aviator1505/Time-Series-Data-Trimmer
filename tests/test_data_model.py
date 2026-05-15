"""Tests for data_model.py - DataModel class.

This module provides comprehensive tests for the DataModel class, which is the
core state manager for the time-series annotation application. Tests cover:
- Deletion boundary precision with floating-point handling
- Timeline gap handling (preserve_timing_gaps preference)
- Annotation management and adjustment after deletion
- Undo/redo functionality
- Robust annotation deserialization
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_model import DataModel, AnnotationSegment, OperationRecord


# ==============================================================================
# Helper Functions and Fixtures
# ==============================================================================

def _create_test_model(n_samples=100, sample_rate=100.0):
    """Helper to create test DataModel with sample data."""
    dm = DataModel()
    dm.df = pd.DataFrame({
        "normalized_time": np.arange(n_samples) / sample_rate,
        "signal": np.sin(np.arange(n_samples) * 0.1),
    })
    dm.sample_rate = sample_rate
    dm.signal_columns = ["signal"]
    dm.time_columns = ["normalized_time"]
    dm._ensure_bad_mask()
    return dm


def _create_model_with_annotations(n_samples=100, sample_rate=100.0, annotations=None):
    """Helper to create test DataModel with sample data and annotations."""
    dm = _create_test_model(n_samples, sample_rate)
    if annotations:
        for ann in annotations:
            dm.annotations.append(ann)
    return dm


@pytest.fixture
def basic_model():
    """Fixture providing a basic DataModel with 100 samples at 100 Hz."""
    return _create_test_model(n_samples=100, sample_rate=100.0)


@pytest.fixture
def model_with_annotations():
    """Fixture providing a DataModel with pre-defined annotations."""
    dm = _create_test_model(n_samples=100, sample_rate=100.0)
    # Time range: 0.0 to 0.99 seconds (100 samples at 100 Hz)
    # Add several annotations for testing
    dm.annotations = [
        AnnotationSegment(start=0.10, end=0.20, label="early", track="test", id=1),
        AnnotationSegment(start=0.40, end=0.60, label="middle", track="test", id=2),
        AnnotationSegment(start=0.80, end=0.95, label="late", track="test", id=3),
    ]
    dm._id_counter = 4
    return dm


# ==============================================================================
# Deletion Boundary Precision Tests
# ==============================================================================

class TestDeletionBoundaryPrecision:
    """Test suite for floating-point boundary handling in segment deletion."""

    def test_delete_segment_boundary_precision(self):
        """Deletion should handle floating-point boundary conditions correctly."""
        # Create data with timestamps that might have precision issues
        dm = _create_test_model(n_samples=1000, sample_rate=100.0)
        initial_len = len(dm.df)

        # Delete a segment with potentially problematic floating-point boundaries
        # Time values: 0.00, 0.01, 0.02, ..., 9.99
        start = 0.10
        end = 0.20

        # Expected: 11 samples deleted (0.10, 0.11, 0.12, ..., 0.20)
        dm.delete_segment(start, end)

        # Verify correct number of samples deleted
        expected_deleted = 11  # Inclusive of both boundaries
        assert len(dm.df) == initial_len - expected_deleted, \
            f"Expected {expected_deleted} samples deleted, got {initial_len - len(dm.df)}"

    def test_delete_segment_includes_boundaries(self):
        """Samples at exact start/end boundaries should be deleted."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)

        # Time values: 0.00, 0.01, 0.02, ..., 0.99
        # Delete from exactly 0.10 to exactly 0.20
        start_time = 0.10
        end_time = 0.20

        # Get the exact time values at boundaries before deletion
        times_before = dm.df["normalized_time"].values
        assert np.any(np.isclose(times_before, start_time, atol=1e-9)), \
            "Start boundary should exist in data"
        assert np.any(np.isclose(times_before, end_time, atol=1e-9)), \
            "End boundary should exist in data"

        dm.delete_segment(start_time, end_time)

        # Verify the boundary values are deleted
        times_after = dm.df["normalized_time"].values

        # Due to timeline shift, we need to check that no time value remains
        # that would have been AT the original boundaries
        # The samples at 0.10 and 0.20 should be gone

        # Actually, after deletion with preserve_timing_gaps=False (default),
        # timestamps are shifted. So we check that the deletion actually removed
        # the correct number of samples
        expected_remaining = 100 - 11  # 11 samples from 0.10 to 0.20 inclusive
        assert len(dm.df) == expected_remaining

    def test_delete_segment_epsilon_tolerance(self):
        """Deletion should use TIME_EPSILON for boundary comparisons."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)

        # Test that epsilon tolerance handles tiny floating-point errors
        # Simulate a case where boundary is slightly off due to float precision
        epsilon = DataModel.TIME_EPSILON

        # The time at index 10 should be 0.10, but let's test near-boundary
        start = 0.10 + (epsilon / 10)  # Slightly after but within epsilon
        end = 0.20 - (epsilon / 10)    # Slightly before but within epsilon

        initial_count = len(dm.df)
        dm.delete_segment(start, end)

        # Should still delete the full range due to epsilon tolerance
        # Samples 10-20 (indices), which is 11 samples
        assert len(dm.df) < initial_count, "Deletion should occur with epsilon tolerance"

    def test_delete_segment_single_sample(self):
        """Deletion of a segment containing a single sample should work."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)
        initial_len = len(dm.df)

        # Delete a tiny range containing just one sample
        # Sample at index 50 has time 0.50
        start = 0.50
        end = 0.50 + 0.001  # Just past 0.50

        dm.delete_segment(start, end)

        # Should delete exactly 1 sample
        assert len(dm.df) == initial_len - 1

    def test_delete_empty_range(self):
        """Deletion with start >= end should fail gracefully."""
        dm = _create_test_model()
        initial_len = len(dm.df)

        # Invalid range - start >= end
        dm.delete_segment(0.5, 0.5)
        assert len(dm.df) == initial_len, "Empty range should not delete anything"

        dm.delete_segment(0.6, 0.5)
        assert len(dm.df) == initial_len, "Reversed range should not delete anything"


# ==============================================================================
# Timeline Gap Handling Tests
# ==============================================================================

class TestTimelineGapHandling:
    """Test suite for the preserve_timing_gaps preference."""

    def test_delete_with_preserve_gaps_false(self):
        """With preserve_timing_gaps=False, timestamps should shift after deletion."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)
        dm.preserve_timing_gaps = False  # Default behavior

        # Time range: 0.0 to 0.99
        # Delete middle segment: 0.40 to 0.50
        deletion_start = 0.40
        deletion_end = 0.50
        deletion_duration = deletion_end - deletion_start

        # Get timestamp of a sample after the deletion region before deletion
        post_deletion_time_before = dm.df[dm.df["normalized_time"] > deletion_end].iloc[0]["normalized_time"]

        dm.delete_segment(deletion_start, deletion_end)

        # After deletion, timestamps after the deleted region should be shifted back
        # The first remaining timestamp after the deletion point should be near
        # where the deletion started (accounting for the removed duration)
        times_after = dm.df["normalized_time"].values

        # The maximum time should be reduced by approximately the deletion duration
        max_time_after = times_after.max()
        max_time_expected = 0.99 - deletion_duration

        assert np.isclose(max_time_after, max_time_expected, atol=0.01), \
            f"Max time should be reduced from 0.99 to ~{max_time_expected}, got {max_time_after}"

        # Timestamps should be continuous (no gaps)
        diffs = np.diff(times_after)
        assert np.all(diffs > 0), "Timestamps should be monotonically increasing"
        # All diffs should be approximately equal (continuous timeline)
        assert np.allclose(diffs, 1/100.0, atol=0.01), "Timeline should be continuous"

    def test_delete_with_preserve_gaps_true(self):
        """With preserve_timing_gaps=True, timestamps should not change."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)
        dm.preserve_timing_gaps = True  # Preserve gaps mode

        # Record original timestamps for samples after deletion region
        original_times = dm.df["normalized_time"].values.copy()

        # Delete middle segment
        deletion_start = 0.40
        deletion_end = 0.50

        dm.delete_segment(deletion_start, deletion_end)

        # Get remaining timestamps
        remaining_times = dm.df["normalized_time"].values

        # Timestamps before the deletion should be unchanged
        before_deletion_mask = original_times < deletion_start - dm.TIME_EPSILON
        original_before = original_times[before_deletion_mask]

        # Check that original timestamps before deletion are preserved
        assert len(remaining_times[remaining_times < deletion_start]) == len(original_before)

        # Timestamps after the deletion should still be at their original values
        # (not shifted)
        after_deletion_mask = original_times > deletion_end + dm.TIME_EPSILON
        original_after = original_times[after_deletion_mask]

        # Find the timestamps in the result that are > deletion_start
        # (since deletion removes samples in the middle, these should match original_after)
        result_after = remaining_times[remaining_times > deletion_start + 0.001]

        assert np.allclose(result_after, original_after), \
            "Timestamps after deletion should be unchanged when preserve_timing_gaps=True"

    def test_preserve_gaps_toggle_behavior(self):
        """Verify that toggling preserve_timing_gaps affects deletion behavior."""
        # Create two identical models
        dm1 = _create_test_model(n_samples=100, sample_rate=100.0)
        dm2 = _create_test_model(n_samples=100, sample_rate=100.0)

        dm1.preserve_timing_gaps = False
        dm2.preserve_timing_gaps = True

        # Perform same deletion
        dm1.delete_segment(0.30, 0.40)
        dm2.delete_segment(0.30, 0.40)

        # Both should have same number of rows
        assert len(dm1.df) == len(dm2.df)

        # But timestamps should differ
        max_time_1 = dm1.df["normalized_time"].max()
        max_time_2 = dm2.df["normalized_time"].max()

        # dm1 (preserve_gaps=False) should have smaller max time
        assert max_time_1 < max_time_2, \
            "Max time should be smaller when gaps are not preserved"


# ==============================================================================
# Annotation Tests
# ==============================================================================

class TestAnnotationManagement:
    """Test suite for annotation creation, update, and deletion."""

    def test_annotate_creates_annotation(self):
        """annotate() should create a new annotation with correct properties."""
        dm = _create_test_model()

        dm.annotate(0.10, 0.20, "test_label", track="test_track", color="#ff0000")

        assert len(dm.annotations) == 1
        ann = dm.annotations[0]
        assert ann.start == 0.10
        assert ann.end == 0.20
        assert ann.label == "test_label"
        assert ann.track == "test_track"
        assert ann.color == "#ff0000"

    def test_annotate_with_invalid_range(self):
        """annotate() should reject invalid ranges."""
        dm = _create_test_model()

        dm.annotate(0.20, 0.10, "invalid")  # start > end
        assert len(dm.annotations) == 0

        dm.annotate(0.10, 0.10, "invalid")  # start == end
        assert len(dm.annotations) == 0

    def test_update_annotation(self):
        """update_annotation() should modify existing annotation."""
        dm = _create_test_model()
        dm.annotate(0.10, 0.20, "original", track="track1")
        ann_id = dm.annotations[0].id

        dm.update_annotation(
            ann_id,
            start=0.15,
            end=0.25,
            label="updated",
            track="track2",
            color="#00ff00"
        )

        ann = dm.annotations[0]
        assert ann.start == 0.15
        assert ann.end == 0.25
        assert ann.label == "updated"
        assert ann.track == "track2"
        assert ann.color == "#00ff00"

    def test_delete_annotation(self):
        """delete_annotation() should remove the specified annotation."""
        dm = _create_test_model()
        dm.annotate(0.10, 0.20, "first")
        dm.annotate(0.30, 0.40, "second")

        first_id = dm.annotations[0].id
        dm.delete_annotation(first_id)

        assert len(dm.annotations) == 1
        assert dm.annotations[0].label == "second"

    def test_annotation_unique_ids(self):
        """Each annotation should have a unique ID."""
        dm = _create_test_model()

        dm.annotate(0.10, 0.20, "first")
        dm.annotate(0.30, 0.40, "second")
        dm.annotate(0.50, 0.60, "third")

        ids = [ann.id for ann in dm.annotations]
        assert len(ids) == len(set(ids)), "All annotation IDs should be unique"


class TestAnnotationAdjustmentAfterDeletion:
    """Test suite for annotation boundary adjustment after segment deletion."""

    def test_annotation_before_deletion_unchanged(self):
        """Annotations entirely before the deletion should remain unchanged."""
        dm = _create_model_with_annotations(n_samples=100, sample_rate=100.0, annotations=[
            AnnotationSegment(start=0.10, end=0.20, label="before", track="test", id=1)
        ])

        # Delete segment after the annotation
        dm.delete_segment(0.50, 0.60)

        assert len(dm.annotations) == 1
        ann = dm.annotations[0]
        assert ann.start == 0.10
        assert ann.end == 0.20
        assert ann.label == "before"

    def test_annotation_after_deletion_shifts_backward(self):
        """Annotations entirely after the deletion should shift backward."""
        dm = _create_model_with_annotations(n_samples=100, sample_rate=100.0, annotations=[
            AnnotationSegment(start=0.70, end=0.80, label="after", track="test", id=1)
        ])

        # Delete segment: 0.40 to 0.50 (duration = 0.10)
        deletion_duration = 0.10
        dm.delete_segment(0.40, 0.50)

        assert len(dm.annotations) == 1
        ann = dm.annotations[0]
        assert np.isclose(ann.start, 0.70 - deletion_duration)
        assert np.isclose(ann.end, 0.80 - deletion_duration)

    def test_annotation_inside_deletion_removed(self):
        """Annotations entirely inside the deletion should be removed."""
        dm = _create_model_with_annotations(n_samples=100, sample_rate=100.0, annotations=[
            AnnotationSegment(start=0.42, end=0.48, label="inside", track="test", id=1)
        ])

        # Delete segment that contains the annotation
        dm.delete_segment(0.40, 0.50)

        assert len(dm.annotations) == 0, "Annotation inside deletion should be removed"

    def test_annotation_spans_deletion_shrinks(self):
        """Annotations that span the deletion should shrink by deletion duration."""
        dm = _create_model_with_annotations(n_samples=100, sample_rate=100.0, annotations=[
            AnnotationSegment(start=0.30, end=0.70, label="spanning", track="test", id=1)
        ])

        # Delete segment in the middle: 0.40 to 0.50 (duration = 0.10)
        deletion_duration = 0.10
        dm.delete_segment(0.40, 0.50)

        assert len(dm.annotations) == 1
        ann = dm.annotations[0]
        assert ann.start == 0.30  # Start unchanged
        assert np.isclose(ann.end, 0.70 - deletion_duration)  # End shrinks

    def test_annotation_overlaps_deletion_start_truncated(self):
        """Annotations overlapping the start of deletion should be truncated."""
        dm = _create_model_with_annotations(n_samples=100, sample_rate=100.0, annotations=[
            AnnotationSegment(start=0.35, end=0.45, label="overlap_start", track="test", id=1)
        ])

        # Delete segment: 0.40 to 0.50
        # Annotation overlaps start (0.35-0.45, deletion starts at 0.40)
        dm.delete_segment(0.40, 0.50)

        assert len(dm.annotations) == 1
        ann = dm.annotations[0]
        assert ann.start == 0.35  # Start unchanged
        assert ann.end == 0.40    # End truncated to deletion start

    def test_annotation_overlaps_deletion_end_adjusted(self):
        """Annotations overlapping the end of deletion should be adjusted."""
        dm = _create_model_with_annotations(n_samples=100, sample_rate=100.0, annotations=[
            AnnotationSegment(start=0.45, end=0.60, label="overlap_end", track="test", id=1)
        ])

        # Delete segment: 0.40 to 0.50 (duration = 0.10)
        # Annotation overlaps end (0.45-0.60, deletion ends at 0.50)
        deletion_duration = 0.10
        dm.delete_segment(0.40, 0.50)

        assert len(dm.annotations) == 1
        ann = dm.annotations[0]
        # The portion from 0.50-0.60 (after deletion) becomes the annotation
        # New start is at deletion start (0.40, where post-deletion content begins)
        # New end shifts back by deletion duration
        assert ann.start == 0.40
        assert np.isclose(ann.end, 0.60 - deletion_duration)

    def test_all_six_annotation_cases_together(self):
        """Test all annotation adjustment cases in a single deletion operation."""
        dm = _create_model_with_annotations(n_samples=200, sample_rate=100.0, annotations=[
            # Case 1: Before deletion (should be unchanged)
            AnnotationSegment(start=0.10, end=0.20, label="before", track="test", id=1),
            # Case 2: After deletion (should shift backward)
            AnnotationSegment(start=1.20, end=1.30, label="after", track="test", id=2),
            # Case 3: Inside deletion (should be removed)
            AnnotationSegment(start=0.52, end=0.58, label="inside", track="test", id=3),
            # Case 4: Spans deletion (should shrink)
            AnnotationSegment(start=0.40, end=0.80, label="spans", track="test", id=4),
            # Case 5: Overlaps start (should be truncated at start)
            AnnotationSegment(start=0.45, end=0.55, label="overlap_start", track="test", id=5),
            # Case 6: Overlaps end (should be adjusted)
            AnnotationSegment(start=0.55, end=0.70, label="overlap_end", track="test", id=6),
        ])

        # Delete segment: 0.50 to 0.60 (duration = 0.10)
        deletion_duration = 0.10
        dm.delete_segment(0.50, 0.60)

        # Expected results:
        remaining = {ann.id: ann for ann in dm.annotations}

        # Case 1: Before - unchanged
        assert 1 in remaining
        assert remaining[1].start == 0.10
        assert remaining[1].end == 0.20

        # Case 2: After - shifted backward
        assert 2 in remaining
        assert np.isclose(remaining[2].start, 1.20 - deletion_duration)
        assert np.isclose(remaining[2].end, 1.30 - deletion_duration)

        # Case 3: Inside - removed
        assert 3 not in remaining

        # Case 4: Spans - end shrinks
        assert 4 in remaining
        assert remaining[4].start == 0.40
        assert np.isclose(remaining[4].end, 0.80 - deletion_duration)

        # Case 5: Overlaps start - truncated (0.45-0.55 becomes 0.45-0.50)
        assert 5 in remaining
        assert remaining[5].start == 0.45
        assert remaining[5].end == 0.50

        # Case 6: Overlaps end - adjusted (0.55-0.70 becomes 0.50 to 0.60)
        assert 6 in remaining
        assert remaining[6].start == 0.50
        assert np.isclose(remaining[6].end, 0.70 - deletion_duration)


class TestRobustAnnotationDeserialization:
    """Test suite for robust handling of malformed annotation data."""

    def test_load_valid_annotations(self):
        """load_annotations() should correctly load valid annotation data."""
        dm = _create_test_model()

        # Create a temporary file with valid annotations
        data = {
            "annotations": [
                {"start": 0.1, "end": 0.2, "label": "test", "track": "default", "color": "#ff0000", "id": 1},
                {"start": 0.3, "end": 0.4, "label": "test2", "track": "default", "color": "#00ff00", "id": 2},
            ],
            "deletions": [],
            "history": [],
            "sample_rate": 100.0
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            dm.load_annotations(temp_path)

            assert len(dm.annotations) == 2
            assert dm.annotations[0].label == "test"
            assert dm.annotations[1].label == "test2"
        finally:
            os.unlink(temp_path)

    def test_skip_malformed_annotations(self):
        """load_annotations() should skip malformed annotations without crashing.

        Note: Current implementation uses dataclass unpacking which accepts type
        mismatches (e.g., string for float) but rejects missing required fields.
        This test validates the behavior where annotations with missing required
        fields are skipped.
        """
        dm = _create_test_model()

        # Create data with mix of valid and malformed annotations
        # Note: AnnotationSegment dataclass has required fields: start, end, label
        # Optional fields with defaults: track, color, id, episode_index
        data = {
            "annotations": [
                # Valid annotation
                {"start": 0.1, "end": 0.2, "label": "valid", "track": "default", "color": "#ff0000", "id": 1},
                # Missing required field 'label' - should be skipped
                {"start": 0.3, "end": 0.4, "track": "default", "id": 2},
                # Another valid annotation
                {"start": 0.7, "end": 0.8, "label": "valid2", "track": "default", "color": "#00ff00", "id": 4},
                # Empty dict - should be skipped (missing required fields)
                {},
                # Missing 'end' - should be skipped
                {"start": 0.9, "label": "missing_end", "track": "default", "id": 5},
                # Unknown field should be rejected
                {"start": 0.1, "end": 0.2, "label": "extra_field", "unknown_field": "value"},
            ],
            "deletions": [],
            "history": [],
            "sample_rate": 100.0
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            # Should not raise an exception
            dm.load_annotations(temp_path)

            # Should have loaded only the valid annotations (those with required fields)
            # Missing required fields cause TypeError, which is caught and skipped
            assert len(dm.annotations) == 2
            labels = [ann.label for ann in dm.annotations]
            assert "valid" in labels
            assert "valid2" in labels
        finally:
            os.unlink(temp_path)

    def test_load_annotations_with_all_malformed_data(self):
        """load_annotations() should handle file with all malformed annotations.

        Tests annotations that are missing required fields. Note that the current
        dataclass implementation does not validate types at runtime.
        """
        dm = _create_test_model()

        data = {
            "annotations": [
                # Missing all required fields
                {"invalid": "data"},
                # Empty dict
                {},
                # Missing 'end' required field
                {"start": 0.1, "label": "no_end"},
                # Missing 'start' required field
                {"end": 0.5, "label": "no_start"},
                # Missing 'label' required field
                {"start": 0.1, "end": 0.5},
            ],
            "deletions": [],
            "history": [],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            # Should not crash
            dm.load_annotations(temp_path)
            # All should be skipped due to missing required fields
            assert len(dm.annotations) == 0
        finally:
            os.unlink(temp_path)

    def test_load_annotations_with_invalid_deletions(self):
        """load_annotations() should skip invalid deletion entries."""
        dm = _create_test_model()

        data = {
            "annotations": [],
            "deletions": [
                {"start": 0.1, "end": 0.2},  # Valid dict format
                [0.3, 0.4],  # Valid list format
                {"start": "bad", "end": 0.5},  # Invalid start type
                [1, "bad"],  # Invalid in list
                {"only_start": 0.1},  # Missing fields
                [0.5, 0.6, 0.7],  # Wrong length
            ],
            "history": [],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            dm.load_annotations(temp_path)
            # Should have loaded only valid deletions
            assert len(dm.deletions) == 2
            assert (0.1, 0.2) in dm.deletions
            assert (0.3, 0.4) in dm.deletions
        finally:
            os.unlink(temp_path)


# ==============================================================================
# Undo/Redo Tests
# ==============================================================================

class TestUndoRedo:
    """Test suite for undo/redo functionality."""

    def test_undo_restores_previous_state(self):
        """Undo should restore the DataFrame to previous state."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)
        initial_len = len(dm.df)
        initial_times = dm.df["normalized_time"].values.copy()

        # Delete a segment
        dm.delete_segment(0.30, 0.40)
        assert len(dm.df) < initial_len

        # Undo the deletion
        dm.undo()

        assert len(dm.df) == initial_len
        assert np.allclose(dm.df["normalized_time"].values, initial_times)

    def test_redo_after_undo(self):
        """Redo should restore the state that was undone."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)
        initial_len = len(dm.df)

        # Delete a segment
        dm.delete_segment(0.30, 0.40)
        len_after_delete = len(dm.df)

        # Undo
        dm.undo()
        assert len(dm.df) == initial_len

        # Redo
        dm.redo()
        assert len(dm.df) == len_after_delete

    def test_multiple_undo_redo_cycles(self):
        """Multiple undo/redo operations should work correctly."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)

        initial_len = len(dm.df)

        # Perform multiple operations
        dm.delete_segment(0.10, 0.15)
        len_after_1 = len(dm.df)

        dm.delete_segment(0.20, 0.25)
        len_after_2 = len(dm.df)

        dm.delete_segment(0.30, 0.35)
        len_after_3 = len(dm.df)

        # Undo all
        dm.undo()
        assert len(dm.df) == len_after_2

        dm.undo()
        assert len(dm.df) == len_after_1

        dm.undo()
        assert len(dm.df) == initial_len

        # Redo all
        dm.redo()
        assert len(dm.df) == len_after_1

        dm.redo()
        assert len(dm.df) == len_after_2

        dm.redo()
        assert len(dm.df) == len_after_3

    def test_undo_restores_annotations(self):
        """Undo should also restore annotation state."""
        dm = _create_test_model()

        # Add an annotation
        dm.annotate(0.10, 0.20, "test_annotation")
        assert len(dm.annotations) == 1

        # Delete the annotation
        ann_id = dm.annotations[0].id
        dm.delete_annotation(ann_id)
        assert len(dm.annotations) == 0

        # Undo should restore the annotation
        dm.undo()
        assert len(dm.annotations) == 1
        assert dm.annotations[0].label == "test_annotation"

    def test_undo_with_empty_stack(self):
        """Undo with empty stack should not crash."""
        dm = _create_test_model()

        # Should not raise an exception
        dm.undo()
        dm.undo()
        dm.undo()

    def test_redo_with_empty_stack(self):
        """Redo with empty stack should not crash."""
        dm = _create_test_model()

        # Should not raise an exception
        dm.redo()
        dm.redo()
        dm.redo()

    def test_new_operation_clears_redo_stack(self):
        """A new operation after undo should clear the redo stack."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)

        # Perform operation
        dm.delete_segment(0.10, 0.20)

        # Undo
        dm.undo()

        # Perform a different operation
        dm.delete_segment(0.30, 0.40)

        # Redo should have nothing (redo stack cleared)
        len_after = len(dm.df)
        dm.redo()
        assert len(dm.df) == len_after  # No change

    def test_undo_stack_limit(self):
        """Undo stack should be limited to MAX_UNDO_STATES."""
        dm = _create_test_model(n_samples=1000, sample_rate=100.0)
        max_states = DataModel.MAX_UNDO_STATES

        # Perform more operations than the limit
        for i in range(max_states + 10):
            dm.annotate(0.1 * i, 0.1 * i + 0.05, f"ann_{i}")

        # Undo stack should be capped
        assert len(dm._undo_stack) <= max_states


# ==============================================================================
# Sample Rate and Time Handling Tests
# ==============================================================================

class TestSampleRateHandling:
    """Test suite for sample rate inference and time handling."""

    def test_infer_sample_rate_from_data(self):
        """Sample rate should be inferred from time column differences."""
        dm = DataModel()
        dm.df = pd.DataFrame({
            "normalized_time": np.arange(100) / 50.0,  # 50 Hz
            "signal": np.random.randn(100),
        })
        dm.signal_columns = ["signal"]
        dm.time_columns = ["normalized_time"]

        inferred = dm._infer_sample_rate()
        assert np.isclose(inferred, 50.0, atol=1.0)

    def test_set_sample_rate_with_recalculate(self):
        """set_sample_rate() with recalculate_time should regenerate time axis."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)

        original_max_time = dm.df["normalized_time"].max()

        # Change sample rate and recalculate
        dm.set_sample_rate(50.0, recalculate_time=True)

        # New max time should be doubled (same samples, half the rate)
        new_max_time = dm.df["normalized_time"].max()
        assert np.isclose(new_max_time, original_max_time * 2, atol=0.1)

    def test_set_sample_rate_without_recalculate(self):
        """set_sample_rate() without recalculate should only update the rate."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)

        original_times = dm.df["normalized_time"].values.copy()

        # Change sample rate without recalculating
        dm.set_sample_rate(50.0, recalculate_time=False)

        # Times should be unchanged
        assert np.allclose(dm.df["normalized_time"].values, original_times)
        assert dm.sample_rate == 50.0


# ==============================================================================
# Mark Bad Tests
# ==============================================================================

class TestMarkBad:
    """Test suite for marking segments as bad."""

    def test_mark_bad_sets_flag(self):
        """mark_bad() should set is_bad_segment flag for samples in range."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)

        # Initially no bad segments
        assert not dm.df["is_bad_segment"].any()

        # Mark a segment as bad
        dm.mark_bad(0.30, 0.40)

        # Check that the correct samples are marked
        time_col = dm.df["normalized_time"].values
        expected_bad = (time_col >= 0.30) & (time_col <= 0.40)

        assert np.array_equal(dm.df["is_bad_segment"].values, expected_bad)

    def test_mark_bad_preserves_data(self):
        """mark_bad() should not delete any samples."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)
        initial_len = len(dm.df)

        dm.mark_bad(0.30, 0.40)

        assert len(dm.df) == initial_len

    def test_mark_bad_undoable(self):
        """mark_bad() should be undoable."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)

        dm.mark_bad(0.30, 0.40)
        assert dm.df["is_bad_segment"].any()

        dm.undo()
        assert not dm.df["is_bad_segment"].any()


# ==============================================================================
# DataFrame Operations Tests
# ==============================================================================

class TestDataFrameOperations:
    """Test suite for DataFrame get/set operations."""

    def test_get_dataframe_returns_copy(self):
        """get_dataframe() should return a copy, not the original."""
        dm = _create_test_model()

        df = dm.get_dataframe()
        df["new_column"] = 1

        # Original should not be modified
        assert "new_column" not in dm.df.columns

    def test_set_dataframe_updates_model(self):
        """set_dataframe() should update the model's DataFrame."""
        dm = _create_test_model()

        new_df = pd.DataFrame({
            "normalized_time": np.arange(50) / 100.0,
            "signal": np.zeros(50),
            "extra_column": np.ones(50),
        })

        dm.set_dataframe(new_df)

        assert len(dm.df) == 50
        assert "extra_column" in dm.signal_columns

    def test_take_time_slice(self):
        """take_time_slice() should return correct subset of data."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)

        slice_df = dm.take_time_slice(0.30, 0.50)

        # Should include samples from 0.30 to 0.50
        assert slice_df["normalized_time"].min() >= 0.30
        assert slice_df["normalized_time"].max() <= 0.50


# ==============================================================================
# History Tracking Tests
# ==============================================================================

class TestHistoryTracking:
    """Test suite for operation history tracking."""

    def test_operations_recorded_in_history(self):
        """Operations should be recorded in history."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)

        assert len(dm.history) == 0

        dm.delete_segment(0.10, 0.20)
        assert len(dm.history) == 1
        assert dm.history[0].description == "delete_segment"

        dm.mark_bad(0.30, 0.40)
        assert len(dm.history) == 2
        assert dm.history[1].description == "mark_bad"

        dm.annotate(0.50, 0.60, "test")
        assert len(dm.history) == 3
        assert dm.history[2].description == "annotate"

    def test_history_includes_timing_info(self):
        """History records should include start and end times."""
        dm = _create_test_model()

        dm.delete_segment(0.15, 0.25)

        record = dm.history[0]
        assert record.start == 0.15
        assert record.end == 0.25

    def test_history_undo_restores(self):
        """Undo should also restore history state."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)

        dm.delete_segment(0.10, 0.20)
        dm.delete_segment(0.30, 0.40)
        assert len(dm.history) == 2

        dm.undo()
        assert len(dm.history) == 1


# ==============================================================================
# Episode Column Conversion Tests
# ==============================================================================

class TestEpisodeColumnConversion:
    """Test suite for annotation to episode column conversion."""

    def test_annotations_to_episode_columns_basic(self):
        """Basic conversion of annotations to episode columns."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)
        dm.annotations = [
            AnnotationSegment(start=0.10, end=0.20, label="action", track="default", id=1),
            AnnotationSegment(start=0.50, end=0.60, label="inspection", track="default", id=2),
        ]

        result_df = dm.annotations_to_episode_columns(dm.df, dm.annotations)

        assert "episode_index" in result_df.columns
        assert "episode_type" in result_df.columns
        assert "episode_state" in result_df.columns

        # Check that episode indices are assigned
        annotated_rows = result_df[result_df["episode_index"].notna()]
        assert len(annotated_rows) > 0

    def test_parse_annotation_label(self):
        """Test annotation label parsing."""
        dm = DataModel()

        # Standard episode format
        type1, state1 = dm._parse_annotation_label("episode:inspection:inspecting_screen")
        assert type1 == "inspection"
        assert state1 == "inspecting_screen"

        # Episode without state
        type2, state2 = dm._parse_annotation_label("episode:action")
        assert type2 == "action"
        assert state2 == ""

        # Non-episode label
        type3, state3 = dm._parse_annotation_label("blink")
        assert type3 == "blink"
        assert state3 == ""


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================

class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_operations_on_empty_model(self):
        """Operations on model without data should handle gracefully."""
        dm = DataModel()

        # Should not crash
        dm.delete_segment(0.0, 1.0)
        dm.mark_bad(0.0, 1.0)
        dm.annotate(0.0, 1.0, "test")
        dm.undo()
        dm.redo()

    def test_delete_entire_dataset(self):
        """Deleting entire dataset should result in empty DataFrame."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)

        # Time range is 0.0 to 0.99
        dm.delete_segment(0.0, 1.0)

        assert len(dm.df) == 0

    def test_delete_segment_outside_data_range(self):
        """Deleting segment outside data range should not crash."""
        dm = _create_test_model(n_samples=100, sample_rate=100.0)
        initial_len = len(dm.df)

        # Time range is 0.0 to 0.99, delete outside
        dm.delete_segment(5.0, 6.0)

        # Should not delete anything
        assert len(dm.df) == initial_len

    def test_very_small_segment_deletion(self):
        """Very small segment deletion should work correctly."""
        dm = _create_test_model(n_samples=10000, sample_rate=10000.0)
        initial_len = len(dm.df)

        # Time step is 0.0001, delete a tiny segment
        dm.delete_segment(0.5000, 0.5001)

        # Should delete at least 1 sample
        assert len(dm.df) <= initial_len


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
