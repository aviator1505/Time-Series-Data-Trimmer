# Copilot Instructions

## Quick start
- Install deps with `pip install -r requirements.txt`; SciPy is optional but unlocks Butterworth filters.
- Launch the GUI from the repo root via `python main.py`; it opens the PyQt6 main window defined in `MainWindow`.
- Use the provided `8_1_P13_Stand_45.csv` as smoke-test data; loading triggers channel classification and UI initialization.

## Architecture highlights
- `main.py` wires together `DataModel`, `FilterEngine`, `ProjectManager`, `PlotController2D/3D`, and Qt dialogs; treat it as the orchestrator.
- User interactions call into `DataModel` for all mutations; the model emits Qt signals that fan back into plotting and history widgets.
- `PlotController2D` renders selected channels and regions; `PlotController3D` visualizes mapped body parts and heading arrows.
- `ProjectManager` tracks multi-trial sessions and recipes, persisting JSON to the path chosen via the project dialogs.
- `PluginManager` discovers JSON recipes in `plugins/`, exposing menu entries under Tools so filters or derived metrics can be replayed.

## Data handling patterns
- `DataModel.load_csv` normalizes NaNs, infers `normalized_time`, and partitions columns into time, metadata, and signals.
- Always include `normalized_time` when supplying new DataFrames; `apply_dataframe` re-classifies columns and records undo history.
- Undo/redo relies on full DataFrame copies; wrap mutations between `_push_state` and signal emissions by calling DataModel helpers instead of editing `df` in place.
- Deletions collapse the timeline and rebuild `normalized_time`; downstream code assumes monotonic seconds.
- Automated suggestions compute spike/NAN segments from the first signal column; keep `signal_columns` ordering sensible when inserting derived channels.

## Filters, plugins, and analytics
- `FilterEngine.apply` operates on channel subsets and honors optional `(start, end)` selections; resampling replaces the entire frame and updates the sample rate.
- When SciPy is missing, Butterworth filters fall back to rolling averages; avoid assuming SciPy-only behavior in new features.
- Recipes and plugins store operations as dicts with `type` (`filter` or `derived`), `channels`, `filter`, and `params`; see `apply_plugin` for supported keys.

## UI coordination
- Selection drags in `PlotController2D` call `set_selection_callback`; maintain that pattern when adding new tools requiring time windows.
- Annotation edits flow through `AnnotationTable` → `data_model.update_annotation`; keep IDs stable so undo/autosave remains consistent.
- 3D views require a mapping from parts to `{x,y,z}` columns; if none is provided, the fallback renders heading-based star plots.

## Projects and persistence
- Project saves store trials, recipes, and preferences; invoke `ProjectManager.save()` after mutating these collections.
- Autosave snapshots live at `.autosave_session.json`; honor this file when changing startup behavior or storage paths.
- Annotation exports (`save_annotations`) persist deletions, history, and sample rate; maintain schema compatibility when extending.

## Developer workflow notes
- No automated tests ship with the repo; validate changes by running the GUI and exercising editing, filtering, and export flows.
- Keep long operations non-blocking by leveraging Qt dialogs (e.g., progress) rather than running heavy loops on the UI thread.
- New dialogs belong in `dialogs.py` to centralize PyQt form patterns and reuse shared styling (legend presets, preset combos, etc.).
- The `plugins/` directory is created on startup; include example JSONs for new functionality so users can reload via Tools → Reload plugins.
