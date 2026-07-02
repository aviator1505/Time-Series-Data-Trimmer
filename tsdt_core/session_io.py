"""Portable session bundles: the .tsdt format.

A .tsdt file is a single ZIP container holding a complete session —
the analog of rerun's .rrd — so work can be moved between machines and
installations as one artifact:

    manifest.json      schema/app version, source name, sample rate
    data.arrow         current DataFrame (Arrow/Feather, zstd-compressed)
    original.arrow     pre-edit DataFrame, enabling revert + replay
    annotations.json   annotations, deletions, operation history
    ui_state.json      optional front-end state (theme, layout, ...)

Arrow preserves dtypes exactly (unlike CSV round-trips), and writes are
atomic (temp file + rename). Newer-major-version bundles are refused
loudly rather than mis-read silently.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from typing import TYPE_CHECKING, Any

import pandas as pd

from tsdt_core.models import AnnotationSegment, OperationRecord

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdt_core.core import CoreDataModel

SESSION_SCHEMA_VERSION = 1
APP_NAME = "time-series-data-trimmer"
SESSION_EXTENSION = ".tsdt"


class Session:
    """Deserialized contents of a .tsdt bundle."""

    def __init__(
        self,
        df: pd.DataFrame,
        original_df: pd.DataFrame | None,
        annotations: list[AnnotationSegment],
        deletions: list[tuple[float, float]],
        history: list[OperationRecord],
        sample_rate: float,
        preserve_timing_gaps: bool,
        ui_state: dict[str, Any],
        manifest: dict[str, Any],
    ) -> None:
        self.df = df
        self.original_df = original_df
        self.annotations = annotations
        self.deletions = deletions
        self.history = history
        self.sample_rate = sample_rate
        self.preserve_timing_gaps = preserve_timing_gaps
        self.ui_state = ui_state
        self.manifest = manifest


def _frame_to_arrow_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.reset_index(drop=True).to_feather(buf, compression="zstd")
    return buf.getvalue()


def _frame_from_arrow_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_feather(io.BytesIO(raw))


def save_session(
    path: str,
    model: CoreDataModel,
    ui_state: dict[str, Any] | None = None,
    source: str | None = None,
) -> None:
    """Write the model's full state to a .tsdt bundle atomically."""
    if model.df is None:
        raise ValueError("No data loaded; nothing to save")

    from tsdt_core import __version__

    manifest = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "app": APP_NAME,
        "app_version": __version__,
        "source": source or "",
        "sample_rate": model.sample_rate,
        "preserve_timing_gaps": model.preserve_timing_gaps,
    }
    annotations_doc = {
        "annotations": [a.model_dump() for a in model.annotations],
        "deletions": [{"start": s, "end": e} for s, e in model.deletions],
        "history": [h.model_dump() for h in model.history],
        "sample_rate": model.sample_rate,
    }

    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, temp_path = tempfile.mkstemp(suffix=SESSION_EXTENSION, dir=out_dir)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            # Arrow payloads are already zstd-compressed; store as-is
            zf.writestr(
                "data.arrow", _frame_to_arrow_bytes(model.df), zipfile.ZIP_STORED
            )
            if model.original_df is not None:
                zf.writestr(
                    "original.arrow",
                    _frame_to_arrow_bytes(model.original_df),
                    zipfile.ZIP_STORED,
                )
            zf.writestr("annotations.json", json.dumps(annotations_doc, indent=2))
            if ui_state:
                zf.writestr("ui_state.json", json.dumps(ui_state, indent=2))
        shutil.move(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def load_session(path: str) -> Session:
    """Read a .tsdt bundle; tolerant of unknown fields, strict on version."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json")) if "manifest.json" in names else {}
        version = int(manifest.get("schema_version", 0))
        if version > SESSION_SCHEMA_VERSION:
            raise ValueError(
                f"Session was written by a newer app version "
                f"(schema {version} > supported {SESSION_SCHEMA_VERSION}); "
                f"please update the application"
            )
        if "data.arrow" not in names:
            raise ValueError("Not a valid .tsdt session: missing data.arrow")

        df = _frame_from_arrow_bytes(zf.read("data.arrow"))
        original_df = (
            _frame_from_arrow_bytes(zf.read("original.arrow"))
            if "original.arrow" in names
            else None
        )
        doc = (
            json.loads(zf.read("annotations.json"))
            if "annotations.json" in names
            else {}
        )
        ui_state = (
            json.loads(zf.read("ui_state.json")) if "ui_state.json" in names else {}
        )

    annotations: list[AnnotationSegment] = []
    for a in doc.get("annotations", []):
        try:
            annotations.append(AnnotationSegment.model_validate(a))
        except (TypeError, ValueError):
            continue
    deletions: list[tuple[float, float]] = []
    for d in doc.get("deletions", []):
        try:
            if isinstance(d, dict):
                deletions.append((float(d["start"]), float(d["end"])))
            else:
                deletions.append((float(d[0]), float(d[1])))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    history: list[OperationRecord] = []
    for h in doc.get("history", []):
        try:
            history.append(OperationRecord.model_validate(h))
        except (TypeError, ValueError):
            continue

    sample_rate = float(manifest.get("sample_rate", doc.get("sample_rate", 120.0)))
    return Session(
        df=df,
        original_df=original_df,
        annotations=annotations,
        deletions=deletions,
        history=history,
        sample_rate=sample_rate,
        preserve_timing_gaps=bool(manifest.get("preserve_timing_gaps", False)),
        ui_state=ui_state,
        manifest=manifest,
    )
