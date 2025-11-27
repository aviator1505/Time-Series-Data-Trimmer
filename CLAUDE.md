# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scientific time-series annotation & cleaning workbench for gaze/kinematics/IMU CSV datasets. PyQt6 GUI application providing interactive segmentation, filtering, annotation, 2D+3D synchronized visualization, and export utilities for research workflows.

## Development Setup

```powershell
# Windows PowerShell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

**Dependencies**: PyQt6, pyqtgraph, pandas, numpy, scipy (optional but recommended for Butterworth filters)

## Running Tests

```powershell
pytest -q
```

Currently only filter engine tests exist at `tests/test_filter_engine.py`. No comprehensive integration tests.

## Architecture

### Component Structure

```
main.py              # Orchestrator: wires DataModel, FilterEngine, ProjectManager,
                     # PlotController2D/3D, dialogs; all Qt signals route through here
data_model.py        # Core state manager: DataFrame wrapper with undo/redo,
                     # annotations, deletion handling, column classification
filter_engine.py     # Signal processing: moving_average, median, savgol, butterworth,
                     # detrend, resample, interpolate, derivative, integrate,
                     # normalize_zscore, normalize_percent, moving_rms, absolute
plot2d.py            # PlotController2D: pyqtgraph-based multichannel time-series
                     # with overlay/stacked modes, selection regions, annotations
plot3d.py            # PlotController3D: pyqtgraph.opengl spatial view with body part
                     # mappings, heading arrows, radial channel fallback
project_manager.py   # Multi-trial sessions: trials table, recipes (saved operation
                     # history), batch processing, preferences persistence
plugin_system.py     # PluginManager: JSON-based operation sequences from plugins/
dialogs.py           # Reusable PyQt6 forms: FilterPanel, AnnotationTable,
                     # MappingDialog, CalibrationWizard, etc.
```

### Data Flow

1. CSV → `DataModel.load_csv()` → NaN normalization + column classification (time/metadata/signals)
2. User operation → `DataModel` mutation method → `_push_state()` for undo snapshot
3. DataFrame update → Qt signal emission → `PlotController2D/3D.refresh()`
4. Operation recorded in history → available for recipe generation
5. Autosave writes `.autosave_session.json` with full state snapshot

### Critical Patterns

**DataModel Mutations**: Always use DataModel helper methods rather than direct DataFrame edits. The model maintains:
- Full DataFrame copies for each undo snapshot (simple but memory-heavy)
- `normalized_time` as monotonically increasing seconds (rebuilt after deletions)
- `is_bad_segment` boolean column for non-destructive marking
- Annotation list with `{id, start, end, label, track, color}` schema

**Column Classification**: On load, columns are partitioned into:
- Time column: `normalized_time` preferred, else first with "time" substring
- Metadata: non-numeric columns
- Signals: numeric columns, further grouped by heuristic (`gaze_*`, `head_*`, `chest_*`, `left_foot_*`, `right_foot_*`, `chair_*`, `gmm_*`, others)

**Deletion Semantics**: Hard deletion collapses timeline via `normalized_time` regeneration (`np.arange(len)/sample_rate`). Mark bad preserves temporal length with boolean flag.

**Filter Operations**: `FilterEngine.apply(df, channels, filter_type, params)` operates on channel subsets. When SciPy missing:
- `savgol` → local polynomial fallback
- `butter_lowpass`/`butter_bandpass` → rolling average approximations

**Plugins & Recipes**:
- Plugin JSON: `{"name": "...", "operations": [{"type": "filter|derived", ...}]}`
- Recipe: captured operation history from interactive session
- Derived channels: `pd.eval` expressions referencing existing columns

## Key Subsystems

### Undo/Redo
Full DataFrame + annotations + deletions + history snapshots pushed before each operation. Memory scales with dataset size × operation count. Use `data_model._push_state()` before mutations.

### Selection & Annotations
- Selection: LinearRegionItem drag in 2D plots → callback to `set_selection_callback`
- Annotation mode: Two single clicks (start, end) → label dialog
- Editing: Double-click annotation row or context menu → `data_model.update_annotation()`
- Keep annotation IDs stable for undo/autosave consistency

### 3D Visualization
Requires body part → `{x, y, z}` column mappings via Tools → 3D mapping. Fallback: star layout from heading values if spatial columns absent. Heading arrows use `<part>_heading_deg` columns.

### Coordinate Frames & Calibration
- Frame transform: Generates relative heading channel: `((source - target - offset + 180) % 360) - 180`
- Calibration wizard: Computes mean heading offset between channels over selected window
- Recorded as `frame_transform` operation in history

### Episode Overlay
Auto-generates annotations from CSV columns: `episode_index`, `episode_type`, optional `episode_state`. Segments rendered with semantic coloring.

### Autosave & Session Recovery
Periodic writes to `.autosave_session.json` containing data (list form), annotations, deletions, history. Restore prompt on startup if file detected.

## Development Constraints

### UI Thread Safety
Heavy operations (filters, resampling) currently run on UI thread. For long operations, wrap in QProgressDialog but consider moving to `QThreadPool` or `QtConcurrent` in future.

### Testing Gaps
Only `tests/test_filter_engine.py` exists. When adding features, manually validate through GUI: load CSV, apply operations, check undo/redo, verify export, test autosave restore.

### SciPy Optional
Always provide fallback implementations for filters. Check `SCIPY_AVAILABLE` flag in `filter_engine.py` before using `scipy.signal` functions.

### Plugin Expression Safety
Derived channel expressions run via `pd.eval` with no sandboxing. Validate JSON schemas but expressions execute with full DataFrame access.

## Common Tasks

### Adding a New Filter
1. Add filter function to `FilterEngine` class
2. Update `available_filters()` list
3. Add to `apply()` method switch statement
4. Provide fallback if SciPy-dependent
5. Add test case to `tests/test_filter_engine.py`

### Adding a New Dialog
1. Subclass `QtWidgets.QDialog` in `dialogs.py`
2. Follow existing patterns (preset combos, legend style)
3. Wire to menu action in `main.py`
4. Emit signals for data updates rather than direct model access

### Extending Annotations
Schema: `{"id", "start", "end", "label", "track", "color"}`. When adding fields:
1. Update `AnnotationSegment` dataclass in `data_model.py`
2. Modify `save_annotations()` and autosave JSON serialization
3. Update `AnnotationTable` columns in `dialogs.py`
4. Maintain backwards compatibility with existing autosave files

### Project & Recipe Changes
When modifying `ProjectManager`:
1. Update `save()` and `load_project()` for schema changes
2. Version the JSON format if breaking compatibility
3. Test batch recipe application with multi-trial workflows

## File Locations

- Autosave: `.autosave_session.json` (root directory)
- Plugins: `plugins/*.json` (auto-created on startup)
- Project files: User-specified `.json` (includes trials, recipes, preferences)
- Test data: `8_1_P13_Stand_45.csv` (smoke test dataset mentioned in docs)

## Performance Notes

- **Memory**: Each undo snapshot = full DataFrame copy. 10 operations on 1M rows × 50 cols ≈ several GB
- **Filtering**: O(N × C) per operation, vectorized via NumPy/pandas
- **Resampling**: O(N) linear interpolation, numeric columns only
- **3D Update**: O(P) where P = body parts + active channel count
- **Suggestions**: O(N) derivative + threshold on first signal channel

For very large datasets (>1M samples), memory pressure from undo stack may require periodic saves + restarts or future diff-based undo implementation.

## Known Limitations

- No packaging/installer; pure script-based distribution
- Limited test coverage beyond filter engine
- UI blocking on heavy operations
- Simple spike detection (derivative threshold); no ML-based artifact detection
- Linear interpolation only for resampling
- No dark mode toggle
- Recipe replay has no validation against changed column schemas
