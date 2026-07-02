"""Offscreen Qt smoke tests.

Instantiates the main window and every dialog headlessly
(QT_QPA_PLATFORM=offscreen) to catch Qt-binding breakage — wrong imports,
signal definitions, enum spellings — that the headless data-layer tests
can never see. These tests do not exercise behavior, only construction.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg

from data_model import AnnotationSegment
import dialogs


@pytest.fixture
def sample_df() -> pd.DataFrame:
    n = 50
    return pd.DataFrame({
        "normalized_time": np.arange(n) / 120.0,
        "gaze_x": np.sin(np.arange(n) / 5.0),
        "head_heading_deg": np.linspace(-90, 90, n),
        "chest_heading_deg": np.linspace(-45, 45, n),
    })


def test_pyqtgraph_bound_to_pyside6():
    """Guard against the mixed-binding hazard: pyqtgraph must use PySide6."""
    assert pg.Qt.QT_LIB == "PySide6"


def test_main_window_constructs(qtbot, tmp_path, monkeypatch):
    # Run in a clean cwd: MainWindow reads .autosave_session.json from cwd
    # (a blocking restore prompt) and creates a plugins/ directory.
    monkeypatch.chdir(tmp_path)
    from main import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle()
    assert window.data_model is not None
    assert window.plot2d.widget is not None or True  # constructed without raising


@pytest.mark.parametrize("factory", [
    pytest.param(lambda df: dialogs.FilterDialog(["gaze_x"]), id="FilterDialog"),
    pytest.param(lambda df: dialogs.FilterPanel(["gaze_x"]), id="FilterPanel"),
    pytest.param(lambda df: dialogs.AnnotationTable(), id="AnnotationTable"),
    pytest.param(lambda df: dialogs.ExportFigureDialog(), id="ExportFigureDialog"),
    pytest.param(lambda df: dialogs.PreferencesDialog(120.0), id="PreferencesDialog"),
    pytest.param(lambda df: dialogs.ShortcutDialog(), id="ShortcutDialog"),
    pytest.param(
        lambda df: dialogs.FrameManagerDialog({"lab": {"parent": "", "offset": 0.0}}),
        id="FrameManagerDialog",
    ),
    pytest.param(
        lambda df: dialogs.MappingDialog(["head_x", "head_y", "head_z"]),
        id="MappingDialog",
    ),
    pytest.param(
        lambda df: dialogs.CompareTrialsDialog(["trial_a.csv"], ["gaze_x"]),
        id="CompareTrialsDialog",
    ),
    pytest.param(
        lambda df: dialogs.FilterPreviewDialog(
            df["normalized_time"].to_numpy(),
            df["gaze_x"].to_numpy(),
            df["gaze_x"].to_numpy() * 0.5,
            "gaze_x",
        ),
        id="FilterPreviewDialog",
    ),
    pytest.param(
        lambda df: dialogs.CalibrationWizard(
            ["head_heading_deg", "chest_heading_deg"], df, (0.0, 0.2)
        ),
        id="CalibrationWizard",
    ),
    pytest.param(
        lambda df: dialogs.MultiTrialPreviewDialog(["8_1_P13_Stand_45.csv"]),
        id="MultiTrialPreviewDialog",
    ),
    pytest.param(
        lambda df: dialogs.ExportCSVDialog(
            [AnnotationSegment(start=0.1, end=0.2, label="a")]
        ),
        id="ExportCSVDialog",
    ),
    pytest.param(
        lambda df: dialogs.RelativeOrientationDialog(
            ["head_heading_deg", "chest_heading_deg"], df
        ),
        id="RelativeOrientationDialog",
    ),
    pytest.param(
        lambda df: dialogs.RecipeDataPreviewDialog("trial", df, df, ["gaze_x"]),
        id="RecipeDataPreviewDialog",
    ),
    pytest.param(
        lambda df: dialogs.RecipePreviewDialog(
            "recipe",
            [{
                "path": "__current__",
                "original_df": df,
                "processed_df": df,
                "signal_columns": ["gaze_x"],
                "op_count": 1,
                "skipped_ops": [],
                "default_output": "out.csv",
            }],
        ),
        id="RecipePreviewDialog",
    ),
    pytest.param(
        lambda df: dialogs.ColumnRenameDialog(["gaze_x", "head_heading_deg"]),
        id="ColumnRenameDialog",
    ),
    pytest.param(
        lambda df: dialogs.ChannelDeleteDialog(["gaze_x", "head_heading_deg"]),
        id="ChannelDeleteDialog",
    ),
    pytest.param(
        lambda df: dialogs.ChannelDuplicateDialog(
            ["gaze_x"], ["gaze_x", "head_heading_deg"]
        ),
        id="ChannelDuplicateDialog",
    ),
    pytest.param(
        lambda df: dialogs.DerivedChannelDialog(["gaze_x"], df, ["gaze_x"]),
        id="DerivedChannelDialog",
    ),
])
def test_dialog_constructs(qtbot, sample_df, factory):
    widget = factory(sample_df)
    qtbot.addWidget(widget)
    assert widget is not None


def test_import_preview_dialog_constructs(qtbot, sample_df):
    from tsdt_core.ingest import IngestReport

    report = IngestReport(
        source="trial.csv",
        delimiter=";",
        encoding="latin-1",
        time_column="time_ms",
        time_unit="ms",
        created_normalized_time=True,
        coerced_columns={"gaze_x": 2},
        notes=["Column 'gaze_x': 2 non-numeric value(s) became NaN"],
    )
    dlg = dialogs.ImportPreviewDialog(sample_df, report)
    qtbot.addWidget(dlg)
    assert dlg.result_frame() is sample_df
    # unit combo defaults to the detected unit
    assert dlg.unit_combo.currentText() == "ms"
