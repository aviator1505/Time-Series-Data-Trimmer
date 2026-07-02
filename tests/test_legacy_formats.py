"""Backward-compatibility guard for on-disk JSON formats.

The fixture files in tests/fixtures/legacy_formats/ were captured from the
serializers as they existed before the Pydantic/schema modernization. These
tests assert that current code can always load sessions written by older
versions of the app. Do NOT regenerate the fixtures from current code — their
whole purpose is to freeze the legacy formats.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_model import DataModel, AnnotationSegment, OperationRecord
from project_manager import ProjectManager

FIXTURES = Path(__file__).parent / "fixtures" / "legacy_formats"


@pytest.fixture
def model_with_data() -> DataModel:
    model = DataModel()
    df = pd.DataFrame({
        "normalized_time": [i / 120.0 for i in range(100)],
        "gaze_x": [0.0] * 100,
    })
    model.set_dataframe(df)
    return model


def test_legacy_annotations_file_loads(model_with_data):
    model_with_data.load_annotations(str(FIXTURES / "annotations_legacy.json"))

    assert len(model_with_data.annotations) == 2
    first, second = model_with_data.annotations
    assert (first.start, first.end, first.label) == (0.1, 0.3, "saccade")
    assert first.track == "gaze"
    assert first.color == "#e15759"
    assert first.id == 101
    assert first.episode_index is None
    assert second.episode_index == 3

    assert model_with_data.deletions == [(0.05, 0.08), (0.7, 0.75)]

    assert len(model_with_data.history) == 2
    op = model_with_data.history[0]
    assert op.description == "filter:butter_lowpass"
    assert op.params["cutoff"] == 6.0
    assert op.params["channels"] == ["gaze_x"]


def test_legacy_project_v1_loads(tmp_path):
    pm = ProjectManager()
    pm.load(str(FIXTURES / "project_v1.json"))

    assert len(pm.trials) == 2
    t = pm.trials[0]
    assert t.path == "8_1_P13_Stand_45.csv"
    assert t.participant == "P13"
    assert t.condition == "Stand"
    assert t.status == "cleaned"
    assert t.trial_number == 8
    assert t.session == 1
    assert t.angle == 45
    # Second trial exercises default values
    assert pm.trials[1].status == "unloaded"

    assert len(pm.recipes) == 1
    recipe = pm.recipes[0]
    assert recipe.name == "standard-clean"
    assert recipe.operations[0]["filter_type"] == "butter_lowpass"
    assert recipe.operations[1]["type"] == "derived"

    # Preferences merge over defaults rather than replacing them
    assert "default_fs" in pm.preferences


def test_legacy_project_saves_after_load(tmp_path):
    """A project loaded from the legacy format must round-trip through save()."""
    pm = ProjectManager()
    pm.load(str(FIXTURES / "project_v1.json"))
    out = tmp_path / "resaved.json"
    pm.project_path = str(out)
    pm.save()

    pm2 = ProjectManager()
    pm2.load(str(out))
    assert [t.path for t in pm2.trials] == [t.path for t in pm.trials]
    assert [r.name for r in pm2.recipes] == [r.name for r in pm.recipes]


def test_legacy_autosave_v2_restores(model_with_data):
    """Replicates the restore path in MainWindow.prompt_restore_autosave."""
    with open(FIXTURES / "autosave_v2.json", encoding="utf-8") as f:
        state = json.load(f)

    assert state["schema_version"] == 2

    df = pd.DataFrame(state["data"])
    assert len(df) == 100
    assert "normalized_time" in df.columns
    assert "gaze_x" in df.columns

    model = model_with_data
    model.set_dataframe(df)

    annotations = [AnnotationSegment(**a) for a in state["annotations"]]
    assert len(annotations) == 2
    assert annotations[0].label == "saccade"

    deletions = [tuple(d) for d in state["deletions"]]
    assert deletions == [(0.05, 0.08), (0.7, 0.75)]

    history = [OperationRecord(**h) for h in state["history"]]
    assert history[0].description == "filter:butter_lowpass"
    assert state["sample_rate"] == 120.0


def test_serialized_keys_match_legacy_format():
    """model_dump() must emit exactly the key sets legacy files contain,
    so files written by this version remain loadable by older versions."""
    from project_manager import Recipe, TrialEntry

    ann = AnnotationSegment(start=0.0, end=1.0, label="x")
    assert set(ann.model_dump()) == {
        "start", "end", "label", "track", "color", "id", "episode_index"
    }

    op = OperationRecord(description="d", params={}, start=0.0, end=1.0)
    assert set(op.model_dump()) == {"description", "params", "start", "end"}

    trial = TrialEntry(path="a.csv")
    assert set(trial.model_dump()) == {
        "path", "participant", "condition", "status", "summary", "notes",
        "trial_number", "session", "angle"
    }

    recipe = Recipe(name="r", operations=[])
    assert set(recipe.model_dump()) == {"name", "operations"}


def test_annotation_coerces_json_types():
    """JSON-sourced values (e.g. ints where floats are expected) must coerce."""
    ann = AnnotationSegment(start=1, end=2, label="x", id=5)
    assert isinstance(ann.start, float)
    assert ann.model_dump()["start"] == 1.0


def test_legacy_annotation_dict_constructs_directly():
    """Raw legacy annotation dicts must construct AnnotationSegment forever."""
    legacy = {
        "start": 1.0,
        "end": 2.0,
        "label": "blink",
        "track": "default",
        "color": "#4e79a7",
        "id": 7,
        "episode_index": None,
    }
    ann = AnnotationSegment(**legacy)
    assert ann.start == 1.0
    assert ann.id == 7

    # Oldest files may omit fields that had defaults
    minimal = {"start": 1.0, "end": 2.0, "label": "blink"}
    ann2 = AnnotationSegment(**minimal)
    assert ann2.track == "default"
    assert ann2.episode_index is None
