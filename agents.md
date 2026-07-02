## Quick start
- Install deps with `pip install -e ".[dev]"` (pinned in `pyproject.toml`); SciPy is optional at runtime but already included, unlocking true Butterworth/Savitzky–Golay filters.
- Launch the GUI from the repo root via `python main.py`; it opens the PySide6 main window defined in `MainWindow`.
- Run tests with `QT_QPA_PLATFORM=offscreen pytest` — the offscreen platform is required for the Qt smoke tests.
- Use the provided `8_1_P13_Stand_45.csv` as smoke-test data; loading triggers adaptive ingestion, channel classification, and UI initialization.

## Architecture highlights
- `tsdt_core/` is the headless engine (no Qt imports allowed there — enforced by a test): `core.py` (`CoreDataModel`), `models.py` (Pydantic v2 models), `undo.py` (diff-based undo entries), `ingest.py` (adaptive file reading), `session_io.py` (`.tsdt` bundles).
- `data_model.py` is a thin Qt adapter: `DataModel(QObject, CoreDataModel)` overrides `_notify_*` hooks to emit signals. Prefer adding new mutation logic to `tsdt_core/core.py`, not `data_model.py`.
- `main.py` wires together `DataModel`, `FilterEngine`, `ProjectManager`, `PlotController2D/3D`, and Qt dialogs; treat it as the orchestrator.
- User interactions call into `DataModel` for all mutations; the model emits Qt signals that fan back into plotting and history widgets.
- `PlotController2D` renders selected channels and regions; `PlotController3D` visualizes mapped body parts and heading arrows.
- `ProjectManager` tracks multi-trial sessions and recipes, persisting JSON to the path chosen via the project dialogs.
- `PluginManager` discovers JSON recipes in `plugins/`, exposing menu entries under Tools so filters or derived metrics can be replayed.
- `background.py` provides `run_in_background()` (QThreadPool + QRunnable) for CSV load and filter application — keep workers pure compute; touch the model/widgets only in the `on_finished` callback on the UI thread.

## Data handling patterns
- `DataModel.load_csv` / `CoreDataModel.load_csv` route through `tsdt_core.ingest.smart_read`, which sniffs delimiter/encoding, detects and unit-converts the time axis into `normalized_time`, and coerces mostly-numeric text columns. Always include `normalized_time` when supplying new DataFrames by hand; `apply_dataframe` re-classifies columns and records undo history.
- Undo/redo is diff-based (`tsdt_core/undo.py`): pick the cheapest sufficient push helper — `_push_meta` (no frame data), `_push_columns`, `_push_rows_removed`, `_push_rename`, or `_push_state` (full snapshot fallback) — and add a round-trip case to `tests/test_undo_diff.py` that checks both undo AND redo restore exact state.
- Deletions collapse the timeline and rebuild `normalized_time`; downstream code assumes monotonic seconds.
- Automated suggestions compute spike/NaN segments from the first signal column; keep `signal_columns` ordering sensible when inserting derived channels.

## Filters, plugins, and analytics
- `FilterEngine.apply` operates on channel subsets and honors optional `(start, end)` selections; it must write only to the channels listed in `params["channels"]` — the column-diff undo path relies on this. Resampling replaces the entire frame (changes length) and takes the full-snapshot undo path.
- When SciPy is missing, Butterworth filters fall back to rolling averages; avoid assuming SciPy-only behavior in new features.
- Recipes and plugins store operations as dicts with `type` (`filter` or `derived`), `channels`, `filter`, and `params`; derived-channel expressions run via `pd.eval` after pattern-based validation (`DANGEROUS_EXPRESSION_PATTERNS` in `main.py`) — see `tests/test_plugin_security.py` before touching this path.

## UI coordination
- Selection drags in `PlotController2D` call `set_selection_callback`; maintain that pattern when adding new tools requiring time windows.
- Annotation edits flow through `AnnotationTable` → `data_model.update_annotation`; keep IDs stable so undo/autosave remains consistent.
- 3D views require a mapping from parts to `{x,y,z}` columns; if none is provided, the fallback renders heading-based star plots.
- Theming goes through `theme.apply_theme(app, "System"|"Light"|"Dark")`; when changing plot styling, restyle via `PlotController2D.set_style(dark=...)` rather than hardcoding colors.

## Projects and persistence
- Project saves store trials, recipes, and preferences (all Pydantic models); invoke `ProjectManager.save()` after mutating these collections.
- `.tsdt` is the native session format (`tsdt_core/session_io.py`): a ZIP of Arrow-encoded data + JSON metadata. Use `CoreDataModel.save_session()`/`load_session()` rather than hand-rolling persistence. Bump `SESSION_SCHEMA_VERSION` on breaking changes; newer-version bundles must be refused on load, not silently misread.
- Autosave writes `.autosave_session.tsdt` every 30s; legacy `.autosave_session.json` files must remain restorable — do not remove that fallback path.
- `tests/fixtures/legacy_formats/` freezes pre-modernization JSON formats; never regenerate them, and keep loading them forever (see `tests/test_legacy_formats.py`).

## Developer workflow notes
- ~300 tests exist; run `QT_QPA_PLATFORM=offscreen pytest` before committing, and add a case for any new mutation, filter, dialog, or ingestion path.
- New dialogs belong in `dialogs.py` and need a construction case in `tests/test_qt_smoke.py` (offscreen instantiation, no PyQt6 — this repo uses PySide6 only).
- Keep long operations non-blocking by routing them through `background.run_in_background` rather than looping on the UI thread.
- The `plugins/` directory is created on startup; include example JSONs for new functionality so users can reload via Tools → Reload plugins.
- Run `ruff check` on `tsdt_core/`, `data_model.py`, `filter_engine.py`, `project_manager.py`, and `plugin_system.py` before committing — CI enforces this on the headless core modules.
