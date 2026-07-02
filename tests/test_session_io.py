"""Round-trip tests for the portable .tsdt session bundle."""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsdt_core import load_session, save_session
from tsdt_core.core import CoreDataModel
from tsdt_core.session_io import SESSION_SCHEMA_VERSION


def make_worked_session(rows: int = 300) -> CoreDataModel:
    """A model with edits of every persisted kind applied."""
    m = CoreDataModel()
    df = pd.DataFrame({
        "normalized_time": np.arange(rows) / 100.0,
        "gaze_x": np.sin(np.arange(rows) / 10.0),
        "head_x": np.cos(np.arange(rows) / 10.0),
        "condition": ["Stand"] * rows,
    })
    m.load_frame(df, "trial.csv")
    m.annotate(0.2, 0.5, "saccade", track="gaze", color="#e15759")
    m.annotate(1.0, 1.4, "fixation")
    m.mark_bad(1.8, 2.0)
    m.delete_segment(2.2, 2.4)
    return m


def test_session_roundtrip_exact(tmp_path):
    m = make_worked_session()
    path = str(tmp_path / "work.tsdt")
    save_session(path, m, ui_state={"theme": "Dark"}, source="trial.csv")

    m2 = CoreDataModel()
    ui_state = m2.load_session(path)

    pd.testing.assert_frame_equal(m2.df, m.df)
    pd.testing.assert_frame_equal(m2.original_df, m.original_df)
    assert [a.model_dump() for a in m2.annotations] == [a.model_dump() for a in m.annotations]
    assert m2.deletions == m.deletions
    assert [h.model_dump() for h in m2.history] == [h.model_dump() for h in m.history]
    assert m2.sample_rate == m.sample_rate
    assert ui_state == {"theme": "Dark"}


def test_session_preserves_dtypes(tmp_path):
    """Arrow must preserve dtypes a CSV round-trip would lose."""
    m = CoreDataModel()
    df = pd.DataFrame({
        "normalized_time": np.arange(10) / 100.0,
        "int_col": np.arange(10, dtype=np.int32),
        "bool_col": [True, False] * 5,
        "float32_col": np.arange(10, dtype=np.float32),
    })
    m.load_frame(df, "t.csv")
    path = str(tmp_path / "dtypes.tsdt")
    save_session(path, m)
    session = load_session(path)
    assert session.df["int_col"].dtype == np.int32
    assert session.df["float32_col"].dtype == np.float32
    assert session.df["bool_col"].dtype == bool


def test_session_bundle_layout(tmp_path):
    m = make_worked_session()
    path = str(tmp_path / "layout.tsdt")
    save_session(path, m, ui_state={"theme": "Dark"})
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        assert {"manifest.json", "data.arrow", "original.arrow",
                "annotations.json", "ui_state.json"} <= names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["schema_version"] == SESSION_SCHEMA_VERSION
        assert manifest["app"] == "time-series-data-trimmer"
        assert manifest["sample_rate"] == m.sample_rate


def test_newer_schema_version_refused(tmp_path):
    m = make_worked_session()
    path = tmp_path / "future.tsdt"
    save_session(str(path), m)
    # Rewrite the manifest to claim a future schema
    with zipfile.ZipFile(path) as zf:
        contents = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(contents["manifest.json"])
    manifest["schema_version"] = SESSION_SCHEMA_VERSION + 1
    contents["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as zf:
        for name, raw in contents.items():
            zf.writestr(name, raw)

    with pytest.raises(ValueError, match="newer app version"):
        load_session(str(path))


def test_invalid_bundle_rejected(tmp_path):
    path = tmp_path / "bogus.tsdt"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError, match="missing data.arrow"):
        load_session(str(path))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_session("/nonexistent/x.tsdt")


def test_save_without_data_raises(tmp_path):
    m = CoreDataModel()
    with pytest.raises(ValueError, match="No data"):
        save_session(str(tmp_path / "empty.tsdt"), m)


def test_load_session_resets_undo_and_continues_editing(tmp_path):
    m = make_worked_session()
    path = str(tmp_path / "cont.tsdt")
    save_session(path, m)

    m2 = CoreDataModel()
    m2.load_session(path)
    assert not m2._undo_stack
    # The restored session must be fully editable
    n_before = len(m2.df)
    m2.delete_segment(0.5, 0.7)
    assert len(m2.df) < n_before
    m2.undo()
    assert len(m2.df) == n_before
    # New annotations get IDs above the restored ones
    m2.annotate(0.1, 0.2, "new")
    ids = [a.id for a in m2.annotations]
    assert len(ids) == len(set(ids))


def test_malformed_annotation_entries_skipped(tmp_path):
    m = make_worked_session()
    path = tmp_path / "malformed.tsdt"
    save_session(str(path), m)
    with zipfile.ZipFile(path) as zf:
        contents = {n: zf.read(n) for n in zf.namelist()}
    doc = json.loads(contents["annotations.json"])
    doc["annotations"].append({"start": 1.0})  # missing required fields
    doc["annotations"].append({"start": 5.0, "end": 6.0, "label": "ok", "future_field": 1})
    contents["annotations.json"] = json.dumps(doc).encode()
    with zipfile.ZipFile(path, "w") as zf:
        for name, raw in contents.items():
            zf.writestr(name, raw)

    session = load_session(str(path))
    labels = [a.label for a in session.annotations]
    assert "ok" in labels           # unknown extra field tolerated
    assert len(session.annotations) == 3  # malformed one skipped
