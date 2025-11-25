# Time-Series Data Trimmer (Kinematics Annotation Studio)

> A scientific time-series annotation & cleaning workbench for gaze / kinematics / IMU CSV datasets featuring no‑code segmentation, filtering, annotation, 2D + 3D synchronized visualization, multi‑trial project management, recipes & plugins, autosave, and export utilities.

---
## Table of Contents
1. Overview
2. Core Concepts & Benefits
3. Feature Matrix
4. Architecture Overview
5. Installation
6. Quick Start (5‑Minute Tour)
7. GUI Walkthrough
8. Data Handling & Column Classification
9. Cleaning Operations (Delete / Mask / Suggestions)
10. Filters & Signal Processing
11. Annotation Workflow
12. 2D Plotting Controller
13. 3D Visualization & Coordinate Frames
14. Projects, Trials, Recipes & Batch Processing
15. Plugins & Derived Channels
16. Autosave & Session Recovery
17. Undo / Redo & Operation History
18. Exporting (Data & Figures)
19. Programmatic API Usage
20. Testing
21. Performance Characteristics
22. Limitations & Transparency
23. Suggested Roadmap / Next Steps
24. Contributing Guidelines
25. License
26. FAQ / Troubleshooting

---
## 1. Overview
This application accelerates exploratory cleaning and annotation of multichannel time‑series signals (e.g., gaze heading, inertial headings, body part positions). It targets researchers who want:
- Rapid manual or semi‑automatic segmentation of artefacts (blinks, spikes, NaNs).
- Consistent annotation of behavioral events (episodes, actions, states).
- Lightweight filtering (smoothing, detrend, normalization, resample) without scripting.
- Visual synchronization of channels in 2D with an optional 3D spatial representation.
- Reproducible batch re‑application of prior cleaning steps via “recipes” or JSON plugins.

You can treat it as a GUI front‑end to a small set of well‑defined data transformation primitives packaged around a pandas DataFrame.

---
## 2. Core Concepts & Benefits
| Concept | Explanation | Benefit |
|---------|-------------|---------|
| `DataModel` | Wraps a pandas DataFrame with undo/redo, annotation state, deletion collapse, classification. | Safe experimentation: full snapshots retained. |
| Normalized Time | Column `normalized_time` maintained as monotonically increasing seconds, regenerated after deletions. | Downstream tools assume contiguous, clean time base. |
| Annotations | Structured segments (`start`, `end`, `label`, `track`, `color`, `id`). | Rich semantic tagging & episode overlay. |
| Deletions vs Bad Segments | Hard removal collapses timeline; marking bad retains samples via flag `is_bad_segment`. | Choose reversible vs irreversible cleaning. |
| Filter Engine | Uniform API for multiple filters with optional SciPy acceleration fallback. | Consistent param surface; preview support. |
| Plugins & Recipes | JSON description of operations or derived expressions. | Reproducibility; batch automation. |
| Project Manager | Aggregates many trial CSV paths plus preferences and recipes. | Multi‑trial study workflow. |
| 2D + 3D Controllers | Linked time cursor & selection; optional mapping to spatial anchors & heading arrows. | Multimodal interpretation of motor/gaze signals. |
| Autosave | Periodic `.autosave_session.json` snapshot of data + annotations + history. | Crash resilience; restore prompt on launch. |
| Suggestions | Simple spike / NaN segment detection on first signal channel. | Speeds manual curation with quick accept. |

---
## 3. Feature Matrix
| Category | Implemented | Notes |
|----------|-------------|-------|
| Load CSV | ✅ | Automatic NaN normalization; classifies columns.
| Column Grouping | ✅ | Heuristic grouping into Gaze, Head, Torso, Feet, Chair, GMM, Other.
| Undo/Redo | ✅ | Full DataFrame copy each operation.
| Selection / Drag | ✅ | Linear region drag; annotation drag updates alive.
| Delete / Mask | ✅ | Delete collapses time; Mark bad sets boolean mask.
| Annotations | ✅ | Color, track, label; editing & context menu.
| Episode Overlay | ✅ | Auto‑generated from columns `episode_index`, `episode_type`, optional `episode_state`.
| Filters | ✅ | Moving average, median, Savitzky–Golay, Butterworth low/band, detrend, resample, interpolate, derivative, integrate, z‑score, percent, moving RMS, absolute.
| Filter Preview | ✅ | First selected channel preview before commit.
| Resampling | ✅ | Linear interpolation, updates sampling rate globally.
| Plugins | ✅ | JSON operations (filter / derived expression).
| Derived Channels | ✅ | Expression via `pd.eval` using existing columns.
| Recipes | ✅ | Saved operation history re‑apply.
| Project Management | ✅ | Trials table; status updates; batch recipe application.
| 2D Plotting | ✅ | Overlay or per‑channel stacked; selection & cursor linking.
| 3D Plotting | ✅ | Mapped (x,y,z) columns; heading arrows; fallback star plot.
| Coordinate Frames | ✅ | Frames with heading offsets; calibration wizard.
| Calibration Wizard | ✅ | Mean heading offset between channels over a window.
| Suggestions (Spike/Nan) | ✅ | Threshold derivative heuristic.
| Autosave | ✅ | Timed + on key events.
| Figure Export | ✅ | PNG/SVG/PDF using pyqtgraph exporters / QPdfWriter.
| Operation History | ✅ | Rolling list with description & param snapshots.
| Keyboard Shortcuts | ✅ | Common operations (D/M/A/U/R/Space/Arrows).
| Testing | Limited | Only filter engine tests included.

---
## 4. Architecture Overview
```text
main.py
  ├─ DataModel (data_model.py)            # Core state, undo/redo, annotations
  ├─ FilterEngine (filter_engine.py)      # Signal transforms
  ├─ PlotController2D (plot2d.py)         # Multi-channel interactive plots
  ├─ PlotController3D (plot3d.py)         # Spatial view & heading arrows
  ├─ PluginManager (plugin_system.py)     # JSON plugin discovery
  ├─ ProjectManager (project_manager.py)  # Multi-trial & recipes
  ├─ Dialogs (dialogs.py)                 # Reusable GUI forms
  └─ Autosave / Suggestions / Playback    # Session resilience & assistance
```
Data flow: CSV → `DataModel` classification → user operations update DataFrame → signals emitted → plot controllers refresh → history recorded → recipes/plugins persist transformations.

Undo/redo: Each mutation uses `_push_state()` to capture a deep copy of DataFrame + annotations + deletions + history before applying changes.

---
## 5. Installation
### Prerequisites
- Python 3.10+ recommended
- Optional: SciPy for real Butterworth & Savitzky–Golay implementations (fallbacks provided otherwise).

### Steps
```powershell
# (Windows PowerShell example)
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```
If SciPy install fails (e.g., no native wheels), the application still runs with graceful fallbacks.

---
## 6. Quick Start (5‑Minute Tour)
1. Launch with `python main.py`.
2. File → Open CSV… (use `8_1_P13_Stand_45.csv` for a smoke test).
3. Toggle channels in the Channel Manager; optionally switch to overlay mode.
4. Drag across a time interval to select; press:
   - `D` delete, `M` mark bad, `A` annotate.
5. Tools → Filters: pick channels & a preset, Preview → Apply.
6. Inspect annotations; edit by double‑clicking or context menu.
7. Enable 3D via Tools → 3D mapping… (enter x,y,z columns).
8. Export cleaned CSV or figure (File → Save / Export figure…).
9. Save annotations & optionally build a recipe from history.
10. Reopen later; autosave offers restoration.

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
| Suggestions Dock | Auto-detected spike/NAN segments to accept as annotations. |
| 2D Plot | Multi-channel time-series (overlay or stacked). Selection & interactive cursor. |
| 3D View | Spatial markers + heading arrows + mirrored channel points. |

Annotation Mode: Click start then end (two single clicks) to create, then label dialog appears.

---
## 8. Data Handling & Column Classification
On load:
- NaN normalization (empty strings, "nan", "NaN" → `np.nan`).
- Time column detection: prefer `normalized_time` else first column containing "time".
- Numeric columns → `signal_columns`; non‑numeric → `metadata_columns`.
- `is_bad_segment` column auto-created if missing.
- Sample rate inferred from median positive time delta; fallback 120 Hz.

Deletion collapses time: After removing a segment, `normalized_time` is reconstructed as `np.arange(n)/sample_rate` ensuring contiguous time base.

Channel groups heuristic string matching (`gaze`, `head`, `chest`, `left_foot`, `right_foot`, `chair`, `gmm`). Unmatched go to `Other`.

---
## 9. Cleaning Operations
- Delete segment: Hard removal + timeline collapse; recorded with `deleted_samples` count.
- Mark bad: Set boolean flag; preserves temporal length.
- Suggestions: Spike detection via derivative threshold mean + 3*std; contiguous indices merged into segments.
- Annotation drag: Updates underlying segment boundaries live.

---
## 10. Filters & Signal Processing
Available filter types (`available_filters()`):
```
moving_average, median, savgol, butter_lowpass, butter_bandpass,
detrend, resample, interpolate, derivative, integrate,
normalize_zscore, normalize_percent, moving_rms, absolute
```
Preview: First selected channel shows side-by-side original vs filtered before committing.

Resample: Linear interpolation of numeric columns to a new uniform grid; updates global sampling rate; non-numeric columns repeat first value.

Fallbacks when SciPy missing:
- Savitzky–Golay: local polynomial fit fallback.
- Butterworth filters: replaced with rolling average or simple bandpass approximation (detrend + lowpass).

Derived metrics (plugins) can use expressions (`pd.eval`) referencing existing columns.

---
## 11. Annotation Workflow
Annotations have: `start`, `end`, `label`, `track`, `color`, `id`.
Context menu: Edit / Delete / Jump.
Episode overlay: If CSV includes `episode_index`, `episode_type`, optional `episode_state`, segments auto-created with semantic coloring.
Drag handles modify boundaries; multi-plot clones stay synchronized.
Autosave ensures annotation persistence.

---
## 12. 2D Plotting Controller
Built on pyqtgraph:
- Overlay vs stacked mode switch.
- LinearRegionItem for selection & annotations.
- Time cursor (InfiniteLine) updated by playback timer or slider.
- Axis styling for readability.
- Focus function pans view to keep a segment centered without zoom change.

---
## 13. 3D Visualization & Coordinate Frames
Uses `pyqtgraph.opengl`:
- Mappings: Body part → `{x,y,z}` column names.
- Each part gets a scatter point + heading arrow (from `<part>_heading_deg`).
- Fallback star layout when spatial columns absent.
- Active 2D channels mirrored as peripheral markers (radial layout, height ~ scaled value).
- Frames: Offsets stored per part; calibration wizard computes mean heading offset between channels over selected window.

### Frame Transform Utility (Tools → Derived frame transform…)
Generates a new heading difference channel between a source and target heading with optional user‑specified offset using modular wrap logic:

```
new = ((source_heading - target_heading - offset + 180) % 360) - 180
```
This keeps angles in the range [-180, 180]. Useful for relative orientation analyses (e.g., head vs torso, gaze vs chair). The new channel is appended to `signal_columns` and recorded in history as `frame_transform`.

---
## 14. Projects, Trials, Recipes & Batch Processing
`ProjectManager` stores:
- Trials: Path, participant, condition, status, summary.
- Recipes: Named sequences of operations.
- Preferences: Default sampling rate, default output directory.

Batch recipe application:
- Load recipe JSON containing operation history entries.
- Reapply filter & plugin operations sequentially across selected trials.
- Output cleaned CSV per trial with `_recipe` suffix.

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
- `derived` operation: `expr` evaluated with `pd.eval`; result appended as new signal column.

Recipes vs Plugins:
- Plugin: static spec of transformations.
- Recipe: captured interactive history (operation records with `description`, `params`, `start`, `end`).

---
## 16. Autosave & Session Recovery
Periodic + event-triggered writes to `.autosave_session.json`, storing:
- Data (list-form per column)
- Annotations
- Deletions
- History
Prompt on startup to restore if file detected.

---
## 17. Undo / Redo & Operation History
Stacks of full state snapshots (DataFrame + annotations + deletions + history).
Pros: Simplicity, reliability.
Cons: Memory-heavy for large datasets → consider a future diff-based model.
History list shows `[description start-end] params` for transparency and recipe generation.

---
## 18. Exporting (Data & Figures)
- Cleaned CSV: File → Save cleaned CSV…
- Annotations: JSON schema with `annotations`, `deletions`, `history`, `sample_rate`.
- Figures: PNG (width scaled by DPI), SVG (vector), PDF (QPdfWriter render of scene).

---
## 19. Programmatic API Usage
You can bypass the GUI for scripted pipelines:
```python
from data_model import DataModel
from filter_engine import FilterEngine
import pandas as pd

model = DataModel()
model.load_csv("trial.csv")
engine = FilterEngine(model.sample_rate)
# Apply smoothing to all signal columns
df = model.get_dataframe()
smoothed = engine.apply(df, model.signal_columns, "savgol", {"window": 11, "polyorder": 2})
model.apply_dataframe(smoothed, "filter", 0.0, smoothed["normalized_time"].max(), {"channels": model.signal_columns, "filter_type": "savgol"})
model.annotate(2.5, 3.0, label="blink", track="eye")
model.save_clean("trial_clean.csv")
model.save_annotations("trial_ann.json")
```
For derived channel creation:
```python
df = model.get_dataframe()
df["gaze_vs_head"] = ((df["gaze_heading_deg"] - df["head_heading_deg"] + 180) % 360) - 180
model.apply_dataframe(df, "derived:gaze_vs_head", 0.0, df["normalized_time"].max(), {})
```

---
## 20. Testing
Currently limited to `tests/test_filter_engine.py` verifying:
- Inclusion of new filter names.
- Correctness of moving RMS and absolute transform.
Run:
```powershell
pytest -q
```
Suggested expansion: add integration tests for deletion, annotation persistence, resample correctness, plugin/recipe replay.

---
## 21. Performance Characteristics
| Aspect | Complexity | Notes |
|--------|------------|-------|
| Filtering | O(N * C) per operation | Rolling windows & interpolation are vectorized.
| Resampling | O(N) | Linear interpolation; numeric only.
| Undo Snapshot | O(N * C) memory | Each action copies full DataFrame (trade-off: simplicity vs memory). |
| Suggestions | O(N) | Derivative & thresholding on first channel.
| 3D Update | O(P) | P body parts + active channels radial markers.

For very large (> millions samples) datasets, memory may spike due to multiple snapshots.

---
## 22. Limitations & Transparency
| Limitation | Detail | Potential Mitigation |
|------------|--------|---------------------|
| Memory Usage | Full DataFrame copies for undo/redo. | Implement diff or delta stacks; compression. |
| Limited Tests | Only filter engine tested. | Expand automated test suite. |
| Plugin Safety | Expressions run with `pd.eval` (no sandbox). | Restrict / validate expressions or run in isolated env. |
| Artefact Detection | Simple derivative threshold & NaN grouping. | ML or statistical multi-channel detectors. |
| UI Blocking | Heavy operations run in UI thread. | Move filters / resample to worker threads with progress dialog. |
| Coordinate Frames | Only heading offset supported. | Hierarchical transforms & quaternion support. |
| Resample Method | Linear interpolation only. | Offer spline / polyphase resampling. |
| No Packaging | Pure script; no installer / wheels. | Add `pyproject.toml`, distribute via PyPI. |
| Missing License | No explicit license file present. | Add OSI-approved license (e.g., MIT). |
| No Dark Mode Toggle | Hard-coded light style in 2D plot. | Theme abstraction & user preference. |
| Recipe Semantics | Blind replay; no validation against changed column set. | Schema versioning & compatibility checks. |

Full transparency: All transformations are in-memory; large chained operations can consume RAM quickly. Deletions irreversibly collapse time (you can undo, but exported data loses original indices).

---
## 23. Suggested Roadmap / Next Steps
1. Testing: Pytest integration across DataModel operations, annotation editing, autosave restore.
2. Performance: Introduce diff-based undo (store changesets) & lazy copy on write.
3. Async Processing: Use `QThreadPool` or `QtConcurrent` for filters & resampling.
4. Advanced Filtering: Add spectral, wavelet, or adaptive filters; configurable parameter presets.
5. Plugin Security: Sandboxed evaluation for derived expressions with allowlist of operations.
6. ML Suggestions: Train classifier for blink/spike/occlusion detection across channels.
7. 3D Enhancements: Camera controls, part hierarchy, smoothing of trajectories, vector fields.
8. Export Improvements: Annotation export to common formats (e.g., CSV, ELAN, JSON-LD). Figure style templates & batch figure export.
9. UI/UX: Dark mode, customizable shortcuts, persistent docking layout.
10. Packaging: Provide `pyproject.toml`, GitHub Actions CI, cross-platform builds, optional installer.
11. Metadata Handling: Rich channel metadata & semantic grouping configuration file.
12. Data Streaming: Support incremental loading & chunked processing of very large trials.
13. Logging/Audit: Structured log of operations for provenance beyond simple history list.

---
## 24. Contributing Guidelines
Proposed (create a CONTRIBUTING.md):
- Fork & branch (`feature/<short-name>`).
- Add/expand tests when touching core logic.
- Keep GUI responsive (avoid long loops on UI thread—prefer vectorized NumPy operations).
- Document new plugins with sample JSON inside `plugins/`.
- Run `pytest` locally before PR.
- Write clear commit messages (imperative mood): "Add moving RMS filter".

---
## 25. License
No license file currently. Until clarified, assume **All Rights Reserved** (limits reuse). Recommendation: adopt MIT or BSD-3-Clause to encourage contribution and academic use.

---
## 26. FAQ / Troubleshooting
| Question | Answer |
|----------|--------|
| SciPy missing errors? | The app falls back to simpler implementations; install SciPy for best results. |
| Why sample rate 120 Hz? | Default when inference fails; can change via Preferences. |
| Deleted segment time jumps? | Timeline is re-collapsed; use mark bad to preserve original duration instead. |
| Large memory usage after many operations? | Each undo snapshot stores full DataFrame; periodically save & restart or await diff-based undo feature. |
| 3D view empty? | Provide mappings via Tools → 3D mapping… referencing x,y,z columns. |
| Plugin not appearing? | Ensure file extension `.json` inside `plugins/` and valid JSON schema with `name`. Reload via Tools → Reload plugins. |
| Filter preview mismatched lengths? | Preview interpolates originals if resample changed time base; truncates on failure. |
| Autosave restore failed? | File may be corrupted; delete `.autosave_session.json` and restart. |

---
## Acknowledgements
Built with PyQt6, pyqtgraph, pandas, numpy, optional SciPy.

---
## Disclaimer
This is a research-oriented tool; not validated for clinical or safety-critical use. Always manually verify transformations before publication.

---
## Citation (Suggested)
If you use this tool in academic work, cite the repository and version tag (add tags/releases in future Roadmap).

---
## Versioning
No formal semantic version yet; recommend adopting `v0.x.y` scheme once roadmap items start landing.

---
Happy annotating & trimming! ✨
