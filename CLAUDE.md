# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scientific time-series annotation & cleaning workbench for gaze/kinematics/IMU datasets. PySide6 GUI application providing interactive segmentation, filtering, annotation, 2D+3D synchronized visualization, portable `.tsdt` session bundles, and export utilities for research workflows. The data engine (`tsdt_core/`) is fully headless and reusable from CLI tools and notebooks.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: . .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python main.py
```

**Dependencies** (pinned in `pyproject.toml`): PySide6-Essentials, pyqtgraph, pandas, numpy, scipy (optional; fallbacks exist), pydantic, pyarrow, charset-normalizer.

## Running Tests

```bash
QT_QPA_PLATFORM=offscreen pytest
```

~300 tests cover the data model, diff-undo invariants, filter engine, adaptive ingestion, session bundles, legacy-format compatibility, plugin security, and offscreen Qt smoke tests (MainWindow + every dialog). CI (`.github/workflows/ci.yml`) runs pytest + ruff on Python 3.11/3.12; `build.yml` produces PyInstaller artifacts per OS.

## Architecture

### Component Structure

```
tsdt_core/           # HEADLESS ENGINE - no Qt imports allowed here
  core.py            #   CoreDataModel: DataFrame wrapper with diff-based
                     #   undo/redo, annotations, deletions, classification;
                     #   observers via overridable _notify_* hooks
  models.py          #   Pydantic v2 models: AnnotationSegment, OperationRecord
  undo.py            #   UndoEntry: per-operation inverse payloads
  ingest.py          #   smart_read(): delimiter/encoding sniffing, time-unit
                     #   detection (epoch s/ms/us/ns, datetimes, name hints),
                     #   numeric coercion with IngestReport
  session_io.py      #   .tsdt bundles: ZIP of Arrow frames + JSON metadata

main.py              # Orchestrator: wires DataModel, FilterEngine, ProjectManager,
                     # PlotController2D/3D, dialogs; all Qt signals route through here
data_model.py        # Thin Qt adapter: DataModel(QObject, CoreDataModel) turns
                     # _notify_* hooks into signals; re-exports the models
background.py        # BackgroundJob/run_in_background: QThreadPool workers with
                     # UI-thread result delivery (used by CSV load, filter apply)
theme.py             # Fusion + System/Light/Dark color schemes, no extra deps
filter_engine.py     # Signal processing: moving_average, median, savgol, butterworth,
                     # detrend, resample, interpolate, derivative, integrate, ...
plot2d.py            # PlotController2D: pyqtgraph multichannel time-series with
                     # overlay/stacked modes, selection regions, annotations
plot3d.py            # PlotController3D: pyqtgraph.opengl spatial view
project_manager.py   # Multi-trial projects: trials table, recipes, preferences
plugin_system.py     # PluginManager: JSON-based operation sequences from plugins/
dialogs.py           # PySide6 forms: FilterPanel, AnnotationTable, MappingDialog,
                     # CalibrationWizard, ImportPreviewDialog, etc.
tsdt.spec            # PyInstaller build definition
tools/benchmark_undo.py  # undo-cost benchmark (run before touching undo paths)
```

### Data Flow

1. File → `smart_read()` on a worker thread → IngestReport (detection/coercion) → optional ImportPreviewDialog → `DataModel.load_frame()` on the UI thread
2. User operation → CoreDataModel mutation method → diff-based undo entry pushed
3. DataFrame update → `_notify_*` hook → Qt signal → `PlotController2D/3D.refresh()`
4. Operation recorded in history → available for recipe generation
5. Autosave writes `.autosave_session.tsdt` (Arrow-based bundle) every 30 s

### Critical Patterns

**Qt/headless boundary**: `tsdt_core/` must never import Qt. UI-facing behavior goes through the `_notify_*` observer hooks; `data_model.DataModel` is the only Qt adapter. A test asserts the core imports without PySide6 loaded.

**Threading**: heavy compute (CSV parse, filter math) runs through `background.run_in_background`; workers must be pure compute on copies — model mutation and widget access happen only in the `on_finished` handler (UI thread).

**Diff-based undo**: each mutation pushes only its inverse payload (see `tsdt_core/undo.py` docstring for kinds). When adding a mutation method, pick the cheapest sufficient push helper: `_push_meta` (no frame data), `_push_columns`, `_push_rows_removed`, `_push_rename`, or `_push_state` (full snapshot fallback). Add a round-trip case to `tests/test_undo_diff.py` — undo AND redo must restore exact state.

**DataModel mutations**: always use model helper methods rather than direct DataFrame edits. The model maintains `normalized_time` (monotonic seconds), `is_bad_segment` (non-destructive marking), and the annotation list.

**Column classification**: on load, columns partition into time / metadata (non-numeric) / signals (numeric), grouped by name heuristics (`gaze_*`, `head_*`, `chest_*`, feet, chair, ...).

**Deletion semantics**: hard deletion collapses the timeline (subsequent timestamps shift back, ms-rounded) unless `preserve_timing_gaps` is set. Mark-bad preserves length with the boolean flag.

**Filter operations**: `FilterEngine.apply(df, channels, filter_type, params)` returns a new frame and writes only to the listed channels (the column-diff undo relies on this — if you add a filter that writes elsewhere, declare it in `params["channels"]` or the undo will corrupt). Resample changes length and takes the full-snapshot path.

**Plugins & recipes**: plugin JSON `{"name", "operations": [...]}`; recipes are captured operation history; derived channels are `pd.eval` expressions (validated by `validate_plugin_expression`, but still full DataFrame access — see plugin security tests).

## Key Subsystems

### Session Bundles (.tsdt)
Single-file ZIP: `manifest.json` (schema/app version, sample rate), `data.arrow` + `original.arrow` (zstd Feather, exact dtypes), `annotations.json`, optional `ui_state.json`. Save/load via `CoreDataModel.save_session/load_session`. Bump `SESSION_SCHEMA_VERSION` on breaking changes; newer-version bundles are refused on load. Legacy JSON autosaves are still restorable.

### Adaptive Ingestion
`tsdt_core/ingest.py` sniffs delimiter/encoding, coerces mostly-numeric string columns (loss counts in `IngestReport`), detects the time axis by name + monotonicity, infers units (epoch magnitude, name hints like `_ms`, datetime strings), and builds `normalized_time` in relative seconds. Files that already have `normalized_time` pass through untouched. `apply_time_axis()` supports explicit user overrides from ImportPreviewDialog. Parquet/Feather also supported.

### Undo/Redo
`UndoEntry.apply(model)` restores the stored state and returns its inverse, making undo/redo symmetric. Memory cap (`MAX_UNDO_MEMORY_MB`) and count cap remain as safety valves; typical entries are KB–MB instead of full-frame copies (see `tools/benchmark_undo.py`).

### Selection & Annotations
- Selection: LinearRegionItem drag in 2D plots → callback to `set_selection_callback`
- Annotation mode: two single clicks (start, end) → label dialog
- Editing: double-click annotation row or context menu → `data_model.update_annotation()`
- Keep annotation IDs stable for undo/autosave consistency

### 3D Visualization
Requires body part → `{x, y, z}` column mappings via Tools → 3D mapping. Fallback: star layout from heading values if spatial columns absent. Heading arrows use `<part>_heading_deg` columns.

### Coordinate Frames & Calibration
- Frame transform: relative heading channel `((source - target - offset + 180) % 360) - 180`
- Calibration wizard computes mean heading offset over a selected window
- Recorded as `frame_transform` operation in history

### Episode Overlay
Auto-generates annotations from CSV columns `episode_index`, `episode_type`, optional `episode_state`.

### Theming
`theme.apply_theme(app, "System"|"Light"|"Dark")` (Preferences → Theme). Uses Qt's color-scheme API with a manual dark palette fallback; returns the effective scheme so plot backgrounds can be restyled (`plot2d.set_style(dark=...)`).

## Development Constraints

### Backward Compatibility
`tests/fixtures/legacy_formats/` freezes the pre-modernization JSON formats (annotations, project v1, autosave v2). These fixtures must never be regenerated; current code must load them forever. Serialized key sets of the Pydantic models are pinned by `test_serialized_keys_match_legacy_format`.

### SciPy Optional
Always provide fallback implementations for filters. Check `SCIPY_AVAILABLE` in `filter_engine.py`.

### Plugin Expression Safety
Derived-channel expressions run via `pd.eval` after pattern-based validation (`DANGEROUS_EXPRESSION_PATTERNS` in main.py); see `tests/test_plugin_security.py` before touching this.

## Common Tasks

### Adding a New Filter
1. Add filter function to `FilterEngine`, update `available_filters()` and `apply()`
2. Write only to the channels passed in (undo relies on it); provide a SciPy fallback
3. Add a test to `tests/test_filter_engine.py`

### Adding a New Dialog
1. Subclass `QtWidgets.QDialog` in `dialogs.py`, wire to a menu action in `main.py`
2. Emit signals for data updates rather than direct model access
3. Add a construction case to `tests/test_qt_smoke.py`

### Extending Annotations
1. Add the field to `AnnotationSegment` in `tsdt_core/models.py` (with a default — old files must load)
2. Update `AnnotationTable` columns in `dialogs.py`
3. Extend the legacy-format tests; serialization is automatic via `model_dump()`

### Extending the Data Model
New mutation methods go in `tsdt_core/core.py` (headless), notify via `_notify_*`, push the cheapest undo entry, and get a round-trip test in `tests/test_undo_diff.py`.

## File Locations

- Autosave: `.autosave_session.tsdt` (cwd; legacy `.autosave_session.json` still restorable)
- Plugins: `plugins/*.json` (auto-created on startup)
- Project files: user-specified `.json`; sessions: user-specified `.tsdt`
- Preferences/UI state: `~/.config/TimeSeriesDataTrimmer/signal_presets.json` (APPDATA on Windows)
- Test data: `8_1_P13_Stand_45.csv` (smoke-test dataset)

## Performance Notes

- **Undo**: diff-based; annotate ~0 cost, filter ≈ one column copy per affected channel, deletion ≈ removed block + time column. Full snapshot only for resample/fallback. Benchmark with `python tools/benchmark_undo.py 1000000 50` before changing undo paths.
- **Filtering**: O(N × affected channels), vectorized; runs off the UI thread
- **Session save**: Arrow/zstd, roughly linear in data size; far faster than the old JSON autosave
- **3D update**: O(P) where P = body parts + active channels

## Known Limitations

- Simple spike detection (derivative threshold); no ML-based artifact detection
- Linear interpolation only for resampling
- Recipe replay has no validation against changed column schemas
- Derived channels/recipes are pandas-expression based (`pd.eval`); keep them on the pandas engine if the internal storage ever changes
