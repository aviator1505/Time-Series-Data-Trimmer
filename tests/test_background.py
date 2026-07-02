"""Tests for the QThreadPool background-job helper."""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from background import run_in_background
from data_model import DataModel


def test_job_delivers_result(qtbot):
    results = []
    job = run_in_background(lambda a, b: a + b, 2, 3, on_finished=results.append)
    qtbot.waitSignal(job.signals.finished, timeout=5000).wait()
    assert results == [5]


def test_job_delivers_error(qtbot):
    errors = []

    def boom():
        raise RuntimeError("worker failed")

    job = run_in_background(boom, on_error=errors.append)
    qtbot.waitSignal(job.signals.error, timeout=5000).wait()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "worker failed"


def test_cancelled_job_emits_nothing(qtbot):
    import threading
    results = []
    release = threading.Event()

    job = run_in_background(lambda: release.wait(5) or 42, on_finished=results.append)
    job.cancel()
    release.set()
    qtbot.wait(300)  # give the pool time to finish the job
    assert results == []


def test_read_csv_frame_matches_load_csv(tmp_path):
    """The split parse/adopt path must behave exactly like load_csv."""
    csv = tmp_path / "t.csv"
    pd.DataFrame({
        "normalized_time": [0.0, 0.01, 0.02, 0.03],
        "gaze_x": [1.0, 2.0, 3.0, 4.0],
        "note": ["", "nan", "ok", "NaN"],
    }).to_csv(csv, index=False)

    via_load_csv = DataModel()
    via_load_csv.load_csv(str(csv))

    via_frames = DataModel()
    df = DataModel.read_csv_frame(str(csv))
    via_frames.load_frame(df, str(csv))

    pd.testing.assert_frame_equal(via_load_csv.df, via_frames.df)
    assert via_load_csv.sample_rate == via_frames.sample_rate
    assert via_load_csv.signal_columns == via_frames.signal_columns
    # NaN normalization happened in the pure-parse step
    assert via_frames.df["note"].isna().sum() == 3


def test_read_csv_frame_missing_file():
    with pytest.raises(FileNotFoundError):
        DataModel.read_csv_frame("/nonexistent/file.csv")
