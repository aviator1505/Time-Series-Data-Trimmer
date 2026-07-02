# Time-Series Data Trimmer (Kinematics Annotation Studio)

> A scientific time-series annotation & cleaning workbench for gaze / kinematics / IMU datasets — no-code segmentation, filtering, annotation, 2D + 3D synchronized visualization, adaptive ingestion, portable `.tsdt` session bundles, multi-trial projects, recipes & plugins, and export utilities.

[![CI](https://img.shields.io/badge/CI-pytest%20%2B%20ruff-informational)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of Contents
1. [Overview](#1-overview)
2. [Core Concepts & Benefits](#2-core-concepts--benefits)
3. [Feature Matrix](#3-feature-matrix)
4. [Architecture](#4-architecture)
5. [Installation](#5-installation)
6. [Quick Start (5-Minute Tour)](#6-quick-start-5-minute-tour)
7. [GUI Walkthrough](#7-gui-walkthrough)
8. [Adaptive Ingestion](#8-adaptive-ingestion)
9. [Cleaning Operations (Delete / Mask / Suggestions)](#9-cleaning-operations)
10. [Filters & Signal Processing](#10-filters--signal-processing)
11. [Annotation Workflow](#11-annotation-workflow)
12. [2D Plotting Controller](#12-2d-plotting-controller)
13. [3D Visualization & Coordinate Frames](#13-3d-visualization--coordinate-frames)
14. [Projects, Trials, Recipes & Batch Processing](#14-projects-trials-recipes--batch-processing)
15. [Plugins & Derived Channels](#15-plugins--derived-channels)
16. [Session Bundles (.tsdt) & Autosave](#16-session-bundles-tsdt--autosave)
17. [Undo / Redo & Operation History](#17-undo--redo--operation-history)
18. [Exporting (Data & Figures)](#18-exporting-data--figures)
19. [Programmatic API Usage](#19-programmatic-api-usage)
20. [Theming](#20-theming)
21. [Testing](#21-testing)
22. [Packaging & Distribution](#22-packaging--distribution)
23. [Performance Characteristics](#23-performance-characteristics)
24. [Limitations & Transparency](#24-limitations--transparency)
25. [Roadmap](#25-roadmap)
26. [Contributing](#26-contributing)
27. [License](#27-license)
28. [FAQ / Troubleshooting](#28-faq--troubleshooting)

---

## 1. Overview

This application accelerates exploratory cleaning and annotation of multichannel time-series signals (e.g., gaze heading, inertial headings, body-part positions). It targets researchers who want:

- Rapid manual or semi-automatic segmentation of artefacts (blinks, spikes, NaNs).
- Consistent annotation of behavioral events (episodes, actions, states).
- Lightweight filtering (smoothing, detrend, normalization, resample) without scripting.
- Visual synchronization of channels in 2D with an optional 3D spatial representation.
- Files that "just load" — delimiter, encoding, and time-unit detection happen automatically instead of requiring a fixed CSV schema.
- Reproducible batch re-application of prior cleaning steps via recipes or JSON plugins.
- Sessions that move between machines as a single portable file.

The GUI is a thin PySide6 layer over `tsdt_core`, a headless engine built on pandas. `tsdt_core` has no Qt dependency and is directly usable from scripts, notebooks, or a future CLI.

---

## 2. Core Concepts & Benefits

| Concept | Explanation | Benefit |
|---------|-------------|---------|
| `CoreDataModel` (`tsdt_core/core.py`) | Wraps a pandas DataFrame with diff-based undo/redo, annotation state, deletion collapse, classification. Qt-free. | Reusable outside the GUI; cheap undo. |
| `DataModel` (`data_model.py`) | Thin `QObject` adapter over `CoreDataModel` — turns its observer hooks into Qt signals. | UI code stays decoupled from data logic. |
| Adaptive ingestion (`tsdt_core/ingest.py`) | Sniffs delimiter/encoding, detects the time axis and its unit, coerces mostly-numeric text columns, reports what it did. | Loads real-world messy exports without manual preprocessing. |
| Normalized Time | Column `normalized_time`, monotonically increasing seconds, rebuilt after deletions. | Downstream tools assume a contiguous, clean time base. |
| Annotations | Structured segments (`start`, `end`, `label`, `track`, `color`, `id`), validated with Pydantic. | Rich semantic tagging & episode overlay. |
| Deletions vs. Bad Segments | Hard removal collapses the timeline; marking bad retains samples via an `is_bad_segment` flag. | Choose reversible vs. irreversible cleaning. |
| Diff-based Undo (`tsdt_core/undo.py`) | Each mutation pushes only its inverse (changed columns, a deleted row block, a rename) — not a full DataFrame copy. | Annotating is near-free; undo stack stays small even on large files. |
| `.tsdt` session bundles | Single ZIP file: current + original data (Arrow/zstd), annotations, history, UI state, versioned manifest. | Portable sessions — move a whole analysis between installations as one file. |
| Filter Engine | Uniform API for many filters with optional SciPy acceleration and pure-NumPy fallbacks. | Consistent parameter surface; preview before commit. |
| Plugins & Recipes | JSON description of operations or derived expressions. | Reproducibility; batch automation. |
| Project Manager | Aggregates many trial CSV paths plus preferences and recipes. | Multi-trial study workflow. |
| 2D + 3D Controllers | Linked time cursor & selection; optional mapping to spatial anchors & heading arrows. | Multimodal interpretation of motor/gaze signals. |
| Background jobs (`background.py`) | CSV parsing and filter application run on a `QThreadPool` worker. | The window stays responsive on large files. |

---

## 3. Feature Matrix

| Category | Implemented | Notes |
|----------|-------------|-------|
| Adaptive file loading | ✅ | Delimiter/encoding sniffing, time-unit detection (epoch s/ms/µs/ns, ISO datetime, name hints), numeric coercion report, Parquet/Feather import. |
| Import preview | ✅ | Shown when ingestion makes a nontrivial decision; lets you override the time column/unit before the file is adopted. |
| Column Grouping | ✅ | Heuristic grouping into Gaze, Head, Torso, Feet, Chair, Workspace, Screen, Fixation, Position, Orientation, Other. |
| Undo/Redo | ✅ | Diff-based — each entry stores only its inverse payload; full snapshot only as a fallback (e.g., resample). |
| Selection / Drag | ✅ | Linear-region drag; annotation drag updates live. |
| Delete / Mask | ✅ | Delete collapses time; Mark Bad sets a boolean mask. |
| Annotations | ✅ | Color, track, label; editing & context menu. |
| Episode Overlay | ✅ | Auto-generated from columns `episode_index`, `episode_type`, optional `episode_state`. |
| Filters | ✅ | Moving average, median, Savitzky–Golay, Butterworth low/band, detrend, resample, interpolate, derivative, integrate, z-score, percent-normalize, moving RMS, absolute, polarity/reference inversion, mirror, circular flip, constant offset. |
| Filter Preview | ✅ | First selected channel preview before commit; runs off the UI thread. |
| Non-blocking I/O | ✅ | CSV load and filter application run on a worker thread via `background.py`. |
| Resampling | ✅ | Linear interpolation; updates sampling rate globally. |
| Plugins | ✅ | JSON operations (filter / derived expression); pattern-validated for safety. |
| Derived Channels | ✅ | Expression via `pd.eval` using existing columns. |
| Recipes | ✅ | Saved operation history, re-applicable across trials. |
| Project Management | ✅ | Trials table; status updates; batch recipe application. |
| 2D Plotting | ✅ | Overlay or per-channel stacked; selection & cursor linking; light/dark backgrounds. |
| 3D Plotting | ✅ | Mapped `(x, y, z)` columns; heading arrows; fallback star plot. |
| Coordinate Frames | ✅ | Frames with heading offsets; calibration wizard. |
| Suggestions (Spike/NaN) | ✅ | Threshold-derivative heuristic. |
| `.tsdt` session bundles | ✅ | Portable single-file save/load (Open/Save session menu, `Ctrl+Shift+S`). |
| Autosave | ✅ | Writes `.autosave_session.tsdt` every 30s; legacy `.autosave_session.json` still restorable. |
| Figure Export | ✅ | PNG/SVG/PDF via pyqtgraph exporters / `QPdfWriter`. |
| Operation History | ✅ | Rolling list with description & param snapshots. |
| Theming | ✅ | System / Light / Dark, no third-party theme dependency. |
| Keyboard Shortcuts | ✅ | `D`/`M`/`A`/`U`/`R`/`Space`/arrows; see Tools → Keyboard Shortcuts in-app. |
| Packaging | ✅ | `pyproject.toml`, PyInstaller spec, per-OS build workflow. |
| Testing | ✅ | ~300 pytest cases: data model, undo invariants, ingestion matrix, session round-trips, plugin security, offscreen Qt smoke tests. |
| CI | ✅ | GitHub Actions: pytest + ruff on Python 3.11/3.12. |

---

## 4. Architecture

```text
tsdt_core/              # HEADLESS ENGINE — no Qt imports allowed
  core.py                 CoreDataModel: DataFrame + diff-undo/redo, annotations,
                           deletions, classification; _notify_* observer hooks
  models.py                Pydantic v2: AnnotationSegment, OperationRecord
  undo.py                   UndoEntry: per-operation inverse payloads
  ingest.py                 smart_read(): delimiter/encoding sniffing, time-unit
                             detection, numeric coercion, IngestReport
  session_io.py              .tsdt bundles: Arrow frames + JSON metadata in a ZIP

main.py                # Orchestrator: wires DataModel, FilterEngine, ProjectManager,
                          PlotController2D/3D, dialogs; Qt signals route through here
data_model.py           # Thin Qt adapter: DataModel(QObject, CoreDataModel)
background.py           # QThreadPool workers for CSV load & filter apply
theme.py                # Fusion + System/Light/Dark color schemes
filter_engine.py        # Signal processing primitives (SciPy + NumPy fallbacks)
plot2d.py               # PlotController2D: pyqtgraph multichannel time-series
plot3d.py               # PlotController3D: pyqtgraph.opengl spatial view
project_manager.py      # Multi-trial projects: trials, recipes, preferences
plugin_system.py        # PluginManager: JSON operation sequences from plugins/
dialogs.py              # PySide6 forms, including ImportPreviewDialog
tsdt.spec               # PyInstaller build definition
tools/benchmark_undo.py # Undo-cost benchmark
```

**Data flow:**
1. File → `smart_read()` on a worker thread → `IngestReport` → optional `ImportPreviewDialog` → `DataModel.load_frame()` on the UI thread.
2. User operation → `CoreDataModel` mutation method → diff-based undo entry pushed.
3. DataFrame update → `_notify_*` hook → Qt signal → `PlotController2D/3D.refresh()`.
4. Operation recorded in history → available for recipe generation.
5. Autosave writes `.autosave_session.tsdt` every 30 seconds.

**Qt/headless boundary:** `tsdt_core/` never imports Qt — a test (`tests/test_undo_diff.py` and friends import it directly) asserts it runs with PySide6 absent from `sys.modules`. All UI-facing behavior flows through the `_notify_*` observer hooks that `data_model.DataModel` overrides to emit signals. This means the entire data engine — ingestion, editing, undo, session I/O — is usable from a plain Python script or a Jupyter notebook with zero GUI dependencies.

---

## 5. Installation

### Prerequisites
- Python 3.11 or 3.12
- Optional: SciPy for the true Butterworth & Savitzky–Golay implementations (pure-NumPy fallbacks are used automatically otherwise — already included via `pyproject.toml`).

### Steps
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: . .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python main.py
```

All dependencies are pinned in `pyproject.toml` (`PySide6-Essentials`, `pyqtgraph`, `pandas`, `numpy`, `scipy`, `pydantic`, `pyarrow`, `charset-normalizer`; dev extras add `pytest`, `pytest-qt`, `ruff`). `requirements.txt` is kept as a quick-start alternative to `pip install -e .`.

On Linux you may need the underlying Qt platform libraries if they aren't already present:
```bash
sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3 libglib2.0-0
```

---

## 6. Quick Start (5-Minute Tour)

1. Launch with `python main.py`.
2. File → Open CSV… (use `8_1_P13_Stand_45.csv` for a smoke test). Ingestion runs in the background; if it had to make a nontrivial decision (odd delimiter, time-unit conversion, coerced columns) you'll see an import preview before the file is adopted.
3. Toggle channels in the Channel Manager; optionally switch to overlay mode.
4. Drag across a time interval to select; press `D` to delete, `M` to mark bad, `A` to annotate.
5. Tools → Filters: pick channels & a preset, Preview → Apply (runs off the UI thread on large files).
6. Inspect annotations; edit by double-clicking or via the context menu.
7. Enable 3D via Tools → 3D mapping… (enter x, y, z columns).
8. File → Save session (`Ctrl+Shift+S`) to write a portable `.tsdt` bundle — or export a cleaned CSV / figure.
9. Build a recipe from the operation history for batch reuse across trials.
10. Reopen later; autosave offers restoration from `.autosave_session.tsdt`.

---

## 7. GUI Walkthrough

| Region | Purpose |
|--------|---------|
| Toolbar | Playback controls, speed, overlay toggle, annotation mode, 3D visibility. |
| Slider | Time cursor scrub (mapped to `normalized_time`). |
| Channel Manager Dock | Checkboxes grouped by heuristic category; save/apply channel presets. |
| Filters Dock | Parameter selection + preview for filter operations. |
| Annotations Dock | Table view of all segments; selecting jumps focus; edit & delete. |
| Operation History Dock | Chronological list of operations with parameter snapshots. |
| Project Dock | Manages multiple trial CSVs with status. |
| Suggestions Dock | Auto-detected spike/NaN segments to accept as annotations. |
| 2D Plot | Multi-channel time-series (overlay or stacked); selection & interactive cursor. |
| 3D View | Spatial markers + heading arrows + mirrored channel points. |

Annotation Mode: click start then end (two single clicks) to create a segment, then a label dialog appears.

---

## 8. Adaptive Ingestion

`tsdt_core/ingest.py` replaces a bare `pd.read_csv` with detection across several axes:

- **Delimiter & encoding**: sniffed via `csv.Sniffer` and `charset-normalizer` from the file's head bytes.
- **Time axis**: detected by column name (`time`, `timestamp`, `datetime`, `clock`, `sec`, `millis`, `micros`) and monotonicity, then converted to `normalized_time` (relative seconds):
  - Epoch magnitude detection distinguishes seconds / milliseconds / microseconds / nanoseconds.
  - Name hints (`_ms`, `_us`, `millis`, `micros`) override magnitude-based guessing.
  - ISO-style datetime strings are parsed and converted to relative seconds.
  - A file that already has `normalized_time` passes through untouched.
- **Numeric coercion**: text columns that are ≥90% numeric are coerced, with a per-column count of values lost, surfaced in the status bar and the import preview.
- **Formats**: CSV/TSV (any sniffed delimiter), Parquet, and Feather.

Everything detected is captured in an `IngestReport` (`tsdt_core/ingest.py`). When ingestion made a nontrivial decision, `ImportPreviewDialog` (`dialogs.py`) shows the report, a data preview, and lets you override the time column/unit before the file is adopted — clean, already-normalized files skip the dialog and load directly.

---

## 9. Cleaning Operations

- **Delete segment**: hard removal + timeline collapse; recorded with a `deleted_samples` count.
- **Mark bad**: sets the `is_bad_segment` boolean flag; preserves temporal length.
- **Suggestions**: spike detection via derivative threshold (mean + 3·std); contiguous indices merged into segments.
- **Annotation drag**: updates underlying segment boundaries live.

---

## 10. Filters & Signal Processing

Available filter types (`filter_engine.available_filters()`):
```
moving_average, median, savgol, butter_lowpass, butter_bandpass,
detrend, resample, interpolate, derivative, integrate,
normalize_zscore, normalize_percent, moving_rms, absolute,
invert_polarity, invert_mean, invert_reference, mirror,
circular_flip, constant_offset
```

Preview: the first selected channel shows original vs. filtered side by side before committing. Filter application runs on a background thread (`background.run_in_background`), so the window stays responsive on large files; a busy dialog's Cancel button detaches the running job.

Resample: linear interpolation of numeric columns to a new uniform grid; updates the global sampling rate; non-numeric columns repeat their first value.

Fallbacks when SciPy is unavailable:
- Savitzky–Golay → local polynomial fit fallback.
- Butterworth filters → rolling-average or detrend+lowpass approximations.

---

## 11. Annotation Workflow

Annotations (`tsdt_core.models.AnnotationSegment`, Pydantic-validated) have: `start`, `end`, `label`, `track`, `color`, `id`, optional `episode_index`. Unknown fields from newer file versions are tolerated rather than rejected.

Context menu: Edit / Delete / Jump. Episode overlay: if a CSV includes `episode_index`, `episode_type`, optional `episode_state`, segments are auto-created with semantic coloring. Drag handles modify boundaries; multi-plot clones stay synchronized.

---

## 12. 2D Plotting Controller

Built on pyqtgraph:
- Overlay vs. stacked mode switch.
- `LinearRegionItem` for selection & annotations.
- Time cursor (`InfiniteLine`) updated by playback timer or slider.
- Light/dark background restyling that follows the active theme.
- Focus function pans the view to keep a segment centered without changing zoom.

---

## 13. 3D Visualization & Coordinate Frames

Uses `pyqtgraph.opengl`:
- Mappings: body part → `{x, y, z}` column names.
- Each part gets a scatter point + heading arrow (from `<part>_heading_deg`).
- Fallback star layout when spatial columns are absent.
- Active 2D channels mirrored as peripheral markers (radial layout, height scaled by value).
- Frames: offsets stored per part; calibration wizard computes the mean heading offset between channels over a selected window.

### Frame Transform Utility (Tools → Derived frame transform…)
Generates a new heading-difference channel between a source and target heading with an optional user-specified offset, wrapped to [-180, 180]:
```
new = ((source_heading - target_heading - offset + 180) % 360) - 180
```
The new channel is appended to `signal_columns` and recorded in history as `frame_transform`.

---

## 14. Projects, Trials, Recipes & Batch Processing

`ProjectManager` (Pydantic-validated `TrialEntry`/`Recipe` models) stores:
- Trials: path, participant, condition, status, summary.
- Recipes: named sequences of operations.
- Preferences: default sampling rate, default output directory.

Batch recipe application loads a recipe JSON, reapplies filter & plugin operations sequentially across selected trials, and writes a cleaned CSV per trial with a `_recipe` suffix.

---

## 15. Plugins & Derived Channels

Directory `plugins/` (auto-created). Each `.json` file may define:
```json
{
  "name": "GazeSmooth",
  "operations": [
    {"type": "filter", "channels": ["gaze_heading_deg"], "filter": "savgol", "params": {"window": 11, "polyorder": 2}},
    {"type": "derived", "name": "gaze_abs", "expr": "abs(gaze_heading_deg)"}
  ]
}
```
- `filter` operation: applied via `FilterEngine.apply`.
- `derived` operation: `expr` evaluated with `pd.eval` after pattern-based validation (`DANGEROUS_EXPRESSION_PATTERNS` in `main.py`); still runs with full DataFrame access — see `tests/test_plugin_security.py`.

Recipes vs. plugins: a plugin is a static spec of transformations; a recipe is captured interactive history (`OperationRecord` entries with `description`, `params`, `start`, `end`).

---

## 16. Session Bundles (.tsdt) & Autosave

`.tsdt` is the native, portable session format — a single ZIP file (`tsdt_core/session_io.py`):

```
manifest.json      schema/app version, source name, sample rate
data.arrow         current DataFrame (Arrow/Feather, zstd-compressed)
original.arrow     pre-edit DataFrame — enables revert & full replay
annotations.json   annotations, deletions, operation history
ui_state.json      optional front-end state (theme, layout, ...)
```

Arrow preserves exact dtypes (unlike a CSV round-trip), writes are atomic (temp file + rename), and bundles from a newer schema version are refused with a clear message rather than silently mis-read. File → Open/Save session (`Ctrl+Shift+S`), or `CoreDataModel.save_session()`/`load_session()` programmatically.

Autosave writes `.autosave_session.tsdt` every 30 seconds and on key events; a restore prompt appears on next launch. Legacy `.autosave_session.json` files from older versions of the app are still detected and restorable.

---

## 17. Undo / Redo & Operation History

Each mutation pushes only its inverse payload (`tsdt_core/undo.py`) instead of a full DataFrame snapshot:

| Mutation | Undo entry stores |
|----------|--------------------|
| Annotate / edit / delete annotation | Nothing but metadata — no frame data. |
| Mark bad | Previous values of the `is_bad_segment` column. |
| Filter apply | Previous values of the affected channels only. |
| Segment delete | The removed contiguous row block + prior time axis. |
| Rename / delete / duplicate / derive channel | Column names, positions, and data needed to reverse the change. |
| Resample (or anything non-diffable) | Full DataFrame snapshot (fallback). |

Applying an entry restores the previous state and returns its inverse, so undo and redo are symmetric. A memory cap (`MAX_UNDO_MEMORY_MB`, default 500 MB) and a count cap (`MAX_UNDO_STATES`, default 30) remain as safety valves, but typical entries are kilobytes to a few megabytes rather than a full-frame copy. Benchmark before touching undo paths:
```bash
python tools/benchmark_undo.py 1000000 50
```
The operation history list shows `[description start–end] params` for transparency and recipe generation.

---

## 18. Exporting (Data & Figures)

- Cleaned CSV: File → Save cleaned CSV… (`Ctrl+S`).
- Session bundle: File → Save session… (`Ctrl+Shift+S`) — the recommended way to preserve full state.
- Annotations: JSON with `annotations`, `deletions`, `history`, `sample_rate` (File → Save annotations…).
- Figures: PNG (DPI-scaled), SVG (vector), PDF (`QPdfWriter` render of the scene).

---

## 19. Programmatic API Usage

`tsdt_core` has no Qt dependency, so you can drive the whole pipeline from a script or notebook:

```python
from tsdt_core import CoreDataModel

model = CoreDataModel()
model.load_csv("trial.csv")          # adaptive ingestion runs automatically

df = model.get_dataframe()
from filter_engine import FilterEngine
engine = FilterEngine(model.sample_rate)
smoothed = engine.apply(df, model.signal_columns, "savgol", {"window": 11, "polyorder": 2})
model.apply_dataframe(
    smoothed, "filter", 0.0, smoothed["normalized_time"].max(),
    {"channels": model.signal_columns, "filter_type": "savgol"},
)

model.annotate(2.5, 3.0, label="blink", track="eye")
model.save_session("trial_clean.tsdt")   # portable single-file save
model.save_clean("trial_clean.csv")      # or a plain CSV export
```

Reload a session headlessly:
```python
model2 = CoreDataModel()
ui_state = model2.load_session("trial_clean.tsdt")
```

For derived-channel creation:
```python
model.create_derived_channel(
    "gaze_vs_head",
    "((gaze_heading_deg - head_heading_deg + 180) % 360) - 180",
)
```

The Qt-based GUI (`data_model.DataModel`) exposes the identical API plus `dataChanged`/`annotationsChanged`/`statusMessage`/`historyChanged` signals.

---

## 20. Theming

`theme.apply_theme(app, "System"|"Light"|"Dark")` — reachable via Edit → Preferences → Theme. Uses Qt's Fusion style plus its native color-scheme API (falling back to a hand-built dark palette on platforms where that API has no effect), so no third-party theming dependency is required. The active scheme also restyles the 2D plot background/foreground (`plot2d.set_style(dark=...)`) and persists across restarts.

---

## 21. Testing

```bash
QT_QPA_PLATFORM=offscreen pytest
```

~300 tests across:
- `test_data_model.py` / `test_undo_diff.py` — deletion boundary precision, annotation adjustment, diff-based undo/redo invariants (every operation kind round-tripped with exact-state equality).
- `test_filter_engine.py` — filter correctness and SciPy-fallback parity.
- `test_ingest.py` — delimiter/encoding/time-unit detection matrix (epoch s/ms/µs, ISO datetimes, dirty delimiters, Latin-1 encoding, numeric coercion).
- `test_session_io.py` — `.tsdt` round-trip, dtype preservation, schema-version rejection, malformed-entry tolerance.
- `test_legacy_formats.py` — pre-modernization JSON formats load forever (frozen fixtures in `tests/fixtures/legacy_formats/`).
- `test_plugin_security.py` — derived-expression sandboxing.
- `test_qt_smoke.py` — offscreen construction of `MainWindow` and all 20+ dialogs, plus a pyqtgraph/PySide6 binding guard.
- `test_background.py`, `test_theme.py` — threading helper and theming behavior.

CI (`.github/workflows/ci.yml`) runs the full suite plus `ruff check` on the headless core modules across Python 3.11 and 3.12.

---

## 22. Packaging & Distribution

- `pyproject.toml` — pinned dependencies, `pip install -e ".[dev]"` for development.
- `tsdt.spec` — PyInstaller build definition (`pyinstaller tsdt.spec`), verified to produce a runnable onedir bundle.
- `.github/workflows/build.yml` — builds Linux/Windows/macOS artifacts on manual dispatch, pushes to `main`, and version tags.

---

## 23. Performance Characteristics

| Aspect | Complexity | Notes |
|--------|------------|-------|
| Filtering | O(N × affected channels) | Vectorized; runs off the UI thread. |
| Resampling | O(N) | Linear interpolation, numeric columns only; full-snapshot undo path. |
| Undo (typical op) | O(diff size) | See the table in [§17](#17-undo--redo--operation-history); annotate ≈ 0, filter ≈ one column copy per affected channel. |
| Undo (fallback) | O(N × C) | Only for non-diffable mutations (resample, unusual scopes). |
| Session save | ~linear in data size | Arrow/zstd; far faster than the old JSON autosave. |
| Suggestions | O(N) | Derivative & thresholding on the first channel. |
| 3D Update | O(P) | P body parts + active channels. |

Measured on 1M rows × 50 columns (`tools/benchmark_undo.py`): annotate dropped from ~3,000 ms / 380 MB to ~0.2 ms; a single-channel filter apply from ~3,300 ms to ~90 ms; the undo stack after 5 mixed operations from 382 MB (holding one usable entry) to ~20 MB (holding all five).

---

## 24. Limitations & Transparency

| Limitation | Detail | Potential Mitigation |
|------------|--------|---------------------|
| Artefact Detection | Simple derivative threshold & NaN grouping. | ML or statistical multi-channel detectors. |
| Resample Method | Linear interpolation only. | Offer spline / polyphase resampling. |
| Coordinate Frames | Only heading offset supported. | Hierarchical transforms & quaternion support. |
| Plugin Safety | Expressions run via `pd.eval` after pattern-based validation, not a real sandbox. | Restrict / validate further or run in an isolated process. |
| Recipe Semantics | Blind replay; no validation against a changed column set. | Schema versioning & compatibility checks. |
| Fallback Undo | Resample and other non-diffable ops still take a full-snapshot cost. | Chunked / lazy snapshotting for these cases. |
| Data Layer | pandas in-memory only; no out-of-core support for very large files. | Optional Polars/Arrow-native storage (see Roadmap). |

Full transparency: all transformations are in-memory; large chained operations still consume RAM proportional to the dataset. Deletions irreversibly collapse time in exported CSVs (undo restores in-session state, but an *exported* cleaned CSV loses original indices — save a `.tsdt` session first if you need to preserve full history).

---

## 25. Roadmap

Tracking the staged modernization plan; items already shipped are noted.

**Shipped**
- PySide6 port, Pydantic v2 models, adaptive ingestion + import preview, non-blocking CSV load/filtering, Qt-native theming, headless `tsdt_core` extraction, diff-based undo, `.tsdt` session bundles, PyInstaller packaging, CI.

**Planned**
1. `rerun-py` export (`File → Export to .rrd`) for shareable, replayable sessions outside this app.
2. Native Polars/Arrow filter execution, with an expression-translation layer so `pd.eval`-based derived channels and recipes keep working.
3. Optional datashader overview strip for >10M-row files (the interactive plot stays on pyqtgraph).
4. `tsdt-batch` CLI built on `tsdt_core` for headless recipe application across a folder of trials.
5. ML-based multi-channel artefact detection to complement the derivative-threshold suggestions.
6. Richer coordinate-frame transforms (hierarchical, quaternion-based).

---

## 26. Contributing

- Branch per change (`feature/<short-name>`); add or extend tests when touching `tsdt_core` or `filter_engine`.
- Keep the Qt/headless boundary intact — no Qt imports in `tsdt_core/`.
- New mutation methods go in `tsdt_core/core.py`, notify via `_notify_*`, push the cheapest sufficient undo entry, and get a round-trip case in `tests/test_undo_diff.py`.
- New dialogs go in `dialogs.py` and get a construction case in `tests/test_qt_smoke.py`.
- Run `QT_QPA_PLATFORM=offscreen pytest` and `ruff check` locally before opening a PR.
- See `CLAUDE.md` for the fuller set of architectural conventions this codebase follows.

---

## 27. License

MIT — see [LICENSE](LICENSE).

---

## 28. FAQ / Troubleshooting

| Question | Answer |
|----------|--------|
| SciPy missing errors? | The app falls back to pure-NumPy implementations; install SciPy (already a pinned dependency) for the true Butterworth/Savitzky–Golay filters. |
| Why sample rate 120 Hz? | Fallback default when inference fails (e.g., too few rows); change it via Preferences or check the import preview for a detected time unit. |
| Deleted segment time jumps? | The timeline is re-collapsed; use Mark Bad instead to preserve original duration. |
| Large memory usage after many operations? | Undo is diff-based, but the fallback path (e.g., resample) still snapshots the full frame; run `tools/benchmark_undo.py` to check your workload. |
| 3D view empty? | Provide mappings via Tools → 3D mapping… referencing x, y, z columns. |
| Plugin not appearing? | Ensure the file has a `.json` extension inside `plugins/` and a valid schema with `name`. Reload via Tools → Reload plugins. |
| Filter preview mismatched lengths? | Preview interpolates originals if resample changed the time base; truncates on failure. |
| Autosave restore failed? | The `.tsdt` bundle may be corrupted or from a newer schema version — delete `.autosave_session.tsdt` and restart. |
| Import preview appeared for a file I expected to load directly? | It only appears when ingestion made a nontrivial decision (odd delimiter/encoding, time-unit conversion, coerced columns) — cancel and fix the source file, or accept the detected settings (they can be overridden in the dialog). |
| Can I use the engine without the GUI? | Yes — `tsdt_core.CoreDataModel` has no Qt dependency; see [§19](#19-programmatic-api-usage). |

---

## Acknowledgements

Built with PySide6, pyqtgraph, pandas, numpy, pydantic, pyarrow, charset-normalizer, and optional SciPy.

## Disclaimer

This is a research-oriented tool; not validated for clinical or safety-critical use. Always manually verify transformations before publication.

---

Happy annotating & trimming! ✨
