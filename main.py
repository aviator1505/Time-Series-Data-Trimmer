"""
Scientific Time-Series Annotation & Cleaning Workbench
------------------------------------------------------

This application targets gaze/kinematics/IMU CSV data and provides
no-code tools for loading, segmenting, annotating, filtering, and
exporting publication-ready figures. It includes a project concept,
undo/redo, 2D+3D synchronized visualization, coordinate frame helpers,
and a minimal plugin/recipe system.

Dependencies
------------
pip install PySide6 pyqtgraph pandas numpy scipy

Run
---
python main.py
"""
from __future__ import annotations

import json
import os
import sys
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class InteractionMode(Enum):
    """Interaction modes for the 2D plot."""
    TRIMMING = auto()     # Selection for data deletion/marking
    ANNOTATION = auto()   # Creating new annotations
    EDIT = auto()         # Editing existing annotations

import re
import shutil
import tempfile

import numpy as np
import pandas as pd
# Import the Qt binding before pyqtgraph so pyqtgraph binds to PySide6
# even if another Qt binding is installed in the environment.
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401 - module provides exporters used dynamically


# Security: Patterns to block in plugin expressions
DANGEROUS_EXPRESSION_PATTERNS = [
    r'\b__\w+__\b',  # Dunder attributes (e.g., __import__, __class__)
    r'\bimport\b',   # Import statements
    r'\bexec\b',     # Exec function
    r'\beval\b',     # Eval function
    r'\bcompile\b',  # Compile function
    r'\bopen\b',     # File operations
    r'\bos\.',       # OS module access
    r'\bsys\.',      # Sys module access
    r'\bsubprocess\.',  # Subprocess module
    r'\bbuiltins\.',    # Builtins access
    r'\bglobals\b',     # Globals access
    r'\blocals\b',      # Locals access
    r'\bgetattr\b',     # Attribute access
    r'\bsetattr\b',     # Attribute setting
    r'\bdelattr\b',     # Attribute deletion
]

# Safe functions allowed in expressions
SAFE_EXPRESSION_FUNCTIONS = {
    'abs', 'sqrt', 'sin', 'cos', 'tan', 'log', 'log10', 'exp', 'pow',
    'mean', 'std', 'min', 'max', 'sum', 'median', 'var',
    'floor', 'ceil', 'round', 'clip',
}


def validate_plugin_expression(expr: str, available_columns: List[str]) -> Tuple[bool, str]:
    """Validate a plugin expression for security.

    Args:
        expr: The expression string to validate
        available_columns: List of available DataFrame column names

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if not expr or not isinstance(expr, str):
        return False, "Expression is empty or not a string"

    # Check for dangerous patterns
    for pattern in DANGEROUS_EXPRESSION_PATTERNS:
        if re.search(pattern, expr, re.IGNORECASE):
            return False, f"Expression contains disallowed pattern: {pattern}"

    # Extract identifiers from expression
    # This matches variable-like tokens
    identifiers = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr))

    # Check each identifier is either a column name or safe function
    for ident in identifiers:
        if ident in available_columns:
            continue
        if ident in SAFE_EXPRESSION_FUNCTIONS:
            continue
        # Allow pandas aggregation keywords
        if ident in {'True', 'False', 'None', 'and', 'or', 'not', 'if', 'else'}:
            continue
        # Allow numpy-style type names in expressions
        if ident in {'float', 'int', 'nan', 'inf'}:
            continue
        return False, f"Unknown identifier '{ident}' - must be a column name or safe function"

    return True, ""

from background import run_in_background
from data_model import AnnotationSegment, DataModel, OperationRecord
from dialogs import (
    AnnotationTable,
    CalibrationWizard,
    ChannelDeleteDialog,
    ChannelDuplicateDialog,
    ColumnRenameDialog,
    CompareTrialsDialog,
    DerivedChannelDialog,
    ExportCSVDialog,
    ExportFigureDialog,
    FilterPanel,
    FilterPreviewDialog,
    FrameManagerDialog,
    MappingDialog,
    PreferencesDialog,
    RecipePreviewDialog,
    RelativeOrientationDialog,
    ShortcutDialog,
)
from filter_engine import FilterEngine
from plot2d import PlotController2D
from plot3d import PlotController3D
from plugin_system import PluginManager
from project_manager import ProjectManager, load_signal_presets, save_signal_presets, load_ui_state, save_ui_state
from theme import apply_theme, effective_scheme


class ChannelManagerWidget(QtWidgets.QWidget):
    """Panel listing time/metadata/signals with show/hide checkboxes."""

    channelToggled = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.style_panel: Optional["ChannelStylePanel"] = None  # set externally
        layout = QtWidgets.QVBoxLayout(self)

        # Create splitter for resizable sections
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Time section
        time_widget = QtWidgets.QWidget()
        time_layout = QtWidgets.QVBoxLayout(time_widget)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.addWidget(QtWidgets.QLabel("Time columns"))
        self.time_list = QtWidgets.QListWidget()
        time_layout.addWidget(self.time_list)
        self.splitter.addWidget(time_widget)

        # Metadata section
        meta_widget = QtWidgets.QWidget()
        meta_layout = QtWidgets.QVBoxLayout(meta_widget)
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.addWidget(QtWidgets.QLabel("Metadata columns"))
        self.meta_list = QtWidgets.QListWidget()
        meta_layout.addWidget(self.meta_list)
        self.splitter.addWidget(meta_widget)

        # Signals section
        signal_widget = QtWidgets.QWidget()
        signal_layout_inner = QtWidgets.QVBoxLayout(signal_widget)
        signal_layout_inner.setContentsMargins(0, 0, 0, 0)
        signal_layout_inner.addWidget(QtWidgets.QLabel("Signals"))
        self.signal_container = QtWidgets.QScrollArea()
        self.signal_container.setWidgetResizable(True)
        self.signal_widget = QtWidgets.QWidget()
        self.signal_layout = QtWidgets.QVBoxLayout(self.signal_widget)
        self.signal_layout.setContentsMargins(0, 0, 0, 0)
        self.signal_container.setWidget(self.signal_widget)
        signal_layout_inner.addWidget(self.signal_container)
        self.splitter.addWidget(signal_widget)

        # Restore saved splitter sizes or use defaults
        ui_state = load_ui_state()
        sizes = ui_state.get("channel_manager_splitter", [80, 80, 300])
        self.splitter.setSizes(sizes)
        self.splitter.splitterMoved.connect(self._save_splitter_state)

        layout.addWidget(self.splitter, 1)

        # Preset controls
        self.presets_combo = QtWidgets.QComboBox()
        self.presets_combo.setEditable(True)
        self.save_preset_btn = QtWidgets.QPushButton("Save")
        self.delete_preset_btn = QtWidgets.QPushButton("Delete")
        p_layout = QtWidgets.QHBoxLayout()
        p_layout.addWidget(self.presets_combo, 1)
        p_layout.addWidget(self.save_preset_btn)
        p_layout.addWidget(self.delete_preset_btn)
        layout.addLayout(p_layout)
        self.presets: Dict[str, Dict] = load_signal_presets()
        self._populate_preset_combo()
        self.save_preset_btn.clicked.connect(self.save_preset)
        self.delete_preset_btn.clicked.connect(self.delete_preset)
        self.presets_combo.currentIndexChanged.connect(self.apply_preset)

    def _save_splitter_state(self) -> None:
        """Save splitter positions to UI state."""
        ui_state = load_ui_state()
        ui_state["channel_manager_splitter"] = self.splitter.sizes()
        save_ui_state(ui_state)

    def populate(self, time_cols: List[str], meta_cols: List[str], signal_cols: Dict[str, List[str]]) -> None:
        self.time_list.clear()
        self.meta_list.clear()
        for c in time_cols:
            self.time_list.addItem(c)
        for c in meta_cols:
            self.meta_list.addItem(c)
        preferred_default = "gaze_heading_deg"
        has_preferred = any(preferred_default in cols for cols in signal_cols.values())
        # clear signals
        for i in reversed(range(self.signal_layout.count())):
            w = self.signal_layout.itemAt(i).widget()
            if w:
                w.setParent(None)
        # grouped signal checkboxes
        for grp, cols in signal_cols.items():
            lbl = QtWidgets.QLabel(f"{grp}")
            lbl.setStyleSheet("font-weight:bold;")
            self.signal_layout.addWidget(lbl)
            for col in cols:
                cb = QtWidgets.QCheckBox(col)
                if has_preferred:
                    cb.setChecked(col == preferred_default)
                else:
                    cb.setChecked(False)
                cb.stateChanged.connect(self.channelToggled)
                self.signal_layout.addWidget(cb)
        self.signal_layout.addStretch(1)

    def get_checked_channels(self) -> List[str]:
        channels: List[str] = []
        for i in range(self.signal_layout.count()):
            w = self.signal_layout.itemAt(i).widget()
            if isinstance(w, QtWidgets.QCheckBox) and w.isChecked():
                channels.append(w.text())
        return channels

    def _populate_preset_combo(self) -> None:
        """Populate the preset combo box from loaded presets."""
        self.presets_combo.blockSignals(True)
        self.presets_combo.clear()
        self.presets_combo.addItem("")  # empty default
        for name in self.presets:
            self.presets_combo.addItem(name)
        self.presets_combo.blockSignals(False)

    def save_preset(self) -> None:
        name = self.presets_combo.currentText().strip()
        if not name:
            return
        channels = self.get_checked_channels()
        styles = self.style_panel.get_all_styles() if self.style_panel else {}
        self.presets[name] = {"channels": channels, "styles": styles}
        save_signal_presets(self.presets)
        if self.presets_combo.findText(name) == -1:
            self.presets_combo.addItem(name)

    def apply_preset(self) -> None:
        name = self.presets_combo.currentText()
        preset = self.presets.get(name)
        if not preset:
            return
        channels = preset.get("channels", [])
        styles = preset.get("styles", {})
        # Apply channel checkboxes
        for i in range(self.signal_layout.count()):
            w = self.signal_layout.itemAt(i).widget()
            if isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(w.text() in channels)
        # Apply styles
        if self.style_panel:
            self.style_panel.set_styles(styles)

    def delete_preset(self) -> None:
        name = self.presets_combo.currentText().strip()
        if not name or name not in self.presets:
            return
        del self.presets[name]
        save_signal_presets(self.presets)
        idx = self.presets_combo.findText(name)
        if idx != -1:
            self.presets_combo.removeItem(idx)


class ChannelStylePanel(QtWidgets.QWidget):
    """Assign per-channel plot styles overriding the global plot style."""

    styleChanged = QtCore.Signal(str, str)

    def __init__(self, style_map: Dict[str, str], parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.style_map = {k: v for k, v in style_map.items() if v in ("line", "scatter", "area")}
        self.styles: Dict[str, str] = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Channel plot styles (default inherits toolbar)"))
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QtWidgets.QWidget()
        self.container_layout = QtWidgets.QFormLayout(self.container)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll, 1)
        layout.addStretch(1)

    def set_channels(self, channels: List[str]) -> None:
        # preserve existing style choices when possible
        existing = dict(self.styles)
        # clear rows
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        for ch in channels:
            combo = QtWidgets.QComboBox()
            combo.addItem("Default", userData="")
            for label, key in self.style_map.items():
                combo.addItem(label, userData=key)
            prev = existing.get(ch, "")
            idx = combo.findData(prev)
            if idx != -1:
                combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(lambda _=0, c=ch, cb=combo: self._on_changed(c, cb))
            self.container_layout.addRow(ch, combo)
        self.styles = {k: v for k, v in existing.items() if k in channels}

    def _on_changed(self, channel: str, combo: QtWidgets.QComboBox) -> None:
        style_key = combo.currentData()
        self.styles[channel] = style_key
        self.styleChanged.emit(channel, style_key)

    def get_all_styles(self) -> Dict[str, str]:
        """Return all non-default style assignments."""
        return {ch: style for ch, style in self.styles.items() if style}

    def set_styles(self, styles: Dict[str, str]) -> None:
        """Apply style mapping from a preset."""
        for i in range(self.container_layout.rowCount()):
            label_item = self.container_layout.itemAt(i, QtWidgets.QFormLayout.ItemRole.LabelRole)
            field_item = self.container_layout.itemAt(i, QtWidgets.QFormLayout.ItemRole.FieldRole)
            if label_item and field_item:
                label_widget = label_item.widget()
                combo = field_item.widget()
                if label_widget and isinstance(combo, QtWidgets.QComboBox):
                    ch = label_widget.text()
                    style = styles.get(ch, "")
                    idx = combo.findData(style)
                    if idx != -1:
                        combo.setCurrentIndex(idx)
                        self.styles[ch] = style


class OperationHistoryWidget(QtWidgets.QListWidget):
    def push(self, text: str) -> None:
        self.addItem(text)
        self.scrollToBottom()


class ProjectPanel(QtWidgets.QWidget):
    trialSelected = QtCore.Signal(str)

    def __init__(self, project: ProjectManager, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.project = project
        layout = QtWidgets.QVBoxLayout(self)
        btns = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add trial")
        self.add_multi_btn = QtWidgets.QPushButton("Add multiple...")
        self.add_folder_btn = QtWidgets.QPushButton("Add folder...")
        self.save_btn = QtWidgets.QPushButton("Save project")
        btns.addWidget(self.add_btn)
        btns.addWidget(self.add_multi_btn)
        btns.addWidget(self.add_folder_btn)
        btns.addWidget(self.save_btn)
        layout.addLayout(btns)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Path", "Participant", "Condition", "Trial", "Session", "Angle", "Status", "Summary"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.add_btn.clicked.connect(self.add_trial)
        self.add_multi_btn.clicked.connect(self.add_trials_multi)
        self.add_folder_btn.clicked.connect(self.add_trials_folder)
        self.save_btn.clicked.connect(self.project.save)
        self.table.cellDoubleClicked.connect(self._emit_selection)

    def refresh(self) -> None:
        self.table.setRowCount(len(self.project.trials))
        for row, t in enumerate(self.project.trials):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(t.path))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(t.participant))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(t.condition))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(t.trial_number) if t.trial_number else ""))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(t.session) if t.session else ""))
            self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(str(t.angle) if t.angle else ""))
            self.table.setItem(row, 6, QtWidgets.QTableWidgetItem(t.status))
            self.table.setItem(row, 7, QtWidgets.QTableWidgetItem(t.summary))

    def add_trial(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Add trial CSV", "", "CSV files (*.csv)")
        if not path:
            return
        participant, _ = QtWidgets.QInputDialog.getText(self, "Participant", "ID (optional)")
        condition, _ = QtWidgets.QInputDialog.getText(self, "Condition", "Condition (optional)")
        self.project.add_trial(path, participant, condition)
        self.refresh()

    def add_trials_multi(self) -> None:
        """Open multi-file dialog and show preview before adding trials."""
        from dialogs import MultiTrialPreviewDialog

        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Trial CSV Files", "", "CSV files (*.csv)"
        )
        if not paths:
            return

        dialog = MultiTrialPreviewDialog(paths, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            entries = dialog.get_trial_entries()
            self.project.add_trials_bulk(entries)
            self.refresh()

    def add_trials_folder(self) -> None:
        """Open folder dialog and add all CSV files within."""
        from dialogs import MultiTrialPreviewDialog
        import glob
        import os

        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Trial CSVs"
        )
        if not folder:
            return

        paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if not paths:
            QtWidgets.QMessageBox.information(
                self, "No Files Found", f"No CSV files found in:\n{folder}"
            )
            return

        dialog = MultiTrialPreviewDialog(paths, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            entries = dialog.get_trial_entries()
            self.project.add_trials_bulk(entries)
            self.refresh()

    def _emit_selection(self, row: int, col: int) -> None:
        if 0 <= row < len(self.project.trials):
            self.trialSelected.emit(self.project.trials[row].path)

    def selected_trials(self) -> List[str]:
        paths: List[str] = []
        for idx in {i.row() for i in self.table.selectedIndexes()}:
            if 0 <= idx < len(self.project.trials):
                paths.append(self.project.trials[idx].path)
        return paths


class GuidedWizard(QtWidgets.QWizard):
    """Simple wizard to walk through basic steps."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Guided Workflow")
        self.addPage(self._page("Step 1: Load file", "Use File → Open CSV to load a trial."))
        self.addPage(self._page("Step 2: Pick channels", "Use the Channel Manager dock to toggle signals."))
        self.addPage(self._page("Step 3: Clean artefacts", "Drag or click to select a segment, then D/M/A."))
        self.addPage(self._page("Step 4: Apply smoothing", "Tools → Filters to smooth gaze/heading."))
        self.addPage(self._page("Step 5: Export", "File → Export cleaned data and figures."))

    def _page(self, title: str, text: str) -> QtWidgets.QWizardPage:
        page = QtWidgets.QWizardPage()
        page.setTitle(title)
        lbl = QtWidgets.QLabel(text)
        lbl.setWordWrap(True)
        lay = QtWidgets.QVBoxLayout(page)
        lay.addWidget(lbl)
        return page


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kinematics Annotation Studio")
        self.resize(1400, 900)
        self.data_model = DataModel()
        self.filter_engine = FilterEngine()
        self.project = ProjectManager()
        self.plugins = PluginManager()
        self.frames: Dict[str, Dict] = {"lab": {"parent": "", "offset": 0.0}}
        self.mapping: Dict[str, Dict[str, str]] = {}
        self.autosave_path = os.path.join(os.getcwd(), ".autosave_session.json")
        self.loaded_file_path: str | None = None
        self.plot2d = PlotController2D()
        self.plot2d.set_style(dark=(effective_scheme(QtWidgets.QApplication.instance()) == "dark"))
        self.plot3d = PlotController3D()
        self.style_panel = ChannelStylePanel(
            {
                "Line": "line",
                "Scatter": "scatter",
                "Area": "area",
            }
        )
        self.plot_style_map = {
            "Line": "line",
            "Scatter": "scatter",
            "Area": "area",
            "Seasonal subseries": "seasonal",
            "Heatmap": "heatmap",
        }
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(40)
        self.autosave_timer = QtCore.QTimer(self)
        self.autosave_timer.setInterval(30000)  # 30 seconds - reduced from 120s for less data loss risk
        self.play_speed = 1.0
        self.current_time = 0.0
        self.snap_to_index = True
        self.playing = False
        self.selection: Tuple[Optional[float], Optional[float]] = (None, None)
        self.interaction_mode = InteractionMode.TRIMMING
        self.last_annotation_label = "event"
        self.suggestion_segments: List[Tuple[float, float, str]] = []
        self.selected_annotation_id: Optional[int] = None
        self._build_ui()
        self._connect_signals()
        self.plugins.load_plugins()
        self.prompt_restore_autosave()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        self.toolbar = QtWidgets.QToolBar("Playback", self)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.prev_action = QtGui.QAction("⏮", self)
        self.prev_action.setToolTip("Step backward")
        self.next_action = QtGui.QAction("⏭", self)
        self.next_action.setToolTip("Step forward")
        self.stop_action = QtGui.QAction("⏹", self)
        self.stop_action.setToolTip("Stop and reset (S)")
        self.stop_action.setShortcut(QtGui.QKeySequence("S"))
        self.play_action = QtGui.QAction("▶/⏸", self)
        # Shortcut handled via explicit QShortcut below to avoid widget-level conflicts
        self.play_action.setToolTip("Play/Pause (Space)")
        self.toolbar.addAction(self.prev_action)
        self.toolbar.addAction(self.play_action)
        self.toolbar.addAction(self.stop_action)
        self.toolbar.addAction(self.next_action)
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentText("1x")
        self.toolbar.addWidget(QtWidgets.QLabel("Speed"))
        self.toolbar.addWidget(self.speed_combo)
        self.overlay_action = QtGui.QAction("Overlay channels", self)
        self.overlay_action.setCheckable(True)
        self.overlay_action.setChecked(False)
        self.toolbar.addAction(self.overlay_action)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QtWidgets.QLabel("Plot style"))
        self.plot_style_combo = QtWidgets.QComboBox()
        self.plot_style_combo.addItems(list(self.plot_style_map.keys()))
        self.plot_style_combo.setCurrentText("Line")
        self.toolbar.addWidget(self.plot_style_combo)
        self.season_label = QtWidgets.QLabel("Period")
        self.season_label.setVisible(False)
        self.season_period_spin = QtWidgets.QDoubleSpinBox()
        self.season_period_spin.setRange(0.01, 1e6)
        self.season_period_spin.setDecimals(2)
        self.season_period_spin.setValue(1.0)
        self.season_period_spin.setSuffix(" s")
        self.season_period_spin.setSingleStep(0.1)
        self.season_period_spin.setVisible(False)
        self.toolbar.addWidget(self.season_label)
        self.toolbar.addWidget(self.season_period_spin)
        # Mode selection buttons (exclusive toggle group)
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)

        self.trim_mode_btn = QtWidgets.QToolButton()
        self.trim_mode_btn.setText("Trim")
        self.trim_mode_btn.setCheckable(True)
        self.trim_mode_btn.setChecked(True)
        self.trim_mode_btn.setToolTip("Selection mode for trimming/deleting data segments (D/M keys)")

        self.annotate_mode_btn = QtWidgets.QToolButton()
        self.annotate_mode_btn.setText("Annotate")
        self.annotate_mode_btn.setCheckable(True)
        self.annotate_mode_btn.setToolTip("Click start/end to create annotations")

        self.edit_mode_btn = QtWidgets.QToolButton()
        self.edit_mode_btn.setText("Edit")
        self.edit_mode_btn.setCheckable(True)
        self.edit_mode_btn.setToolTip("Drag annotations to adjust start/end times")

        self.mode_group.addButton(self.trim_mode_btn, InteractionMode.TRIMMING.value)
        self.mode_group.addButton(self.annotate_mode_btn, InteractionMode.ANNOTATION.value)
        self.mode_group.addButton(self.edit_mode_btn, InteractionMode.EDIT.value)

        self.toolbar.addWidget(self.trim_mode_btn)
        self.toolbar.addWidget(self.annotate_mode_btn)
        self.toolbar.addWidget(self.edit_mode_btn)
        self.toolbar.addSeparator()
        self.show_3d_action = QtGui.QAction("Show 3D", self)
        self.show_3d_action.setCheckable(True)
        self.show_3d_action.setChecked(False)
        self.toolbar.addAction(self.show_3d_action)
        self.show_annotations_action = QtGui.QAction("Annotations", self)
        self.show_annotations_action.setCheckable(True)
        self.show_annotations_action.setChecked(True)
        self.show_annotations_action.setToolTip("Show/hide annotation overlays on plot")
        self.toolbar.addAction(self.show_annotations_action)
        self.cursor_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.cursor_slider.setRange(0, 1000)
        layout.addWidget(self.cursor_slider)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.plot2d.widget)
        # 3D container with placeholder text
        self.gl_container = QtWidgets.QWidget()
        gl_layout = QtWidgets.QVBoxLayout(self.gl_container)
        gl_layout.setContentsMargins(0, 0, 0, 0)
        placeholder = QtWidgets.QLabel("3D view (enable via Tools → 3D mapping)")
        placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #888;")
        gl_layout.addWidget(placeholder)
        gl_layout.addWidget(self.plot3d.view, 1)
        self.gl_container.setVisible(False)
        self.splitter.addWidget(self.gl_container)
        self.splitter.setSizes([1200, 1])
        layout.addWidget(self.splitter, 1)
        self.channel_manager = ChannelManagerWidget()
        self.ann_table = AnnotationTable()
        self.history_widget = OperationHistoryWidget()
        self.project_panel = ProjectPanel(self.project)
        self.suggestions = QtWidgets.QListWidget()
        self.suggestions.setMaximumHeight(120)
        self.filter_panel = FilterPanel(self.data_model.signal_columns, self)
        self.ann_table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.ann_table.customContextMenuRequested.connect(self._show_annotation_menu)
        self.filter_dock = self._add_dock("Filters", self.filter_panel, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
        self.style_dock = self._add_dock("Plot styles", self.style_panel, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
        self._add_dock("Channel Manager", self.channel_manager, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
        self.channel_manager.style_panel = self.style_panel
        self._add_dock("Annotations", self.ann_table, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self._add_dock("Operation History", self.history_widget, QtCore.Qt.DockWidgetArea.BottomDockWidgetArea)
        self._add_dock("Project", self.project_panel, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
        self._add_dock("Suggestions", self.suggestions, QtCore.Qt.DockWidgetArea.BottomDockWidgetArea)
        self.snap_index_chk = QtWidgets.QCheckBox("Snap to index")
        self.snap_index_chk.setChecked(True)
        self.snap_peak_chk = QtWidgets.QCheckBox("Snap to extremum")
        self.statusBar().addPermanentWidget(self.snap_index_chk)
        self.statusBar().addPermanentWidget(self.snap_peak_chk)
        self._build_menus()

    def _add_dock(self, title: str, widget: QtWidgets.QWidget, area: QtCore.Qt.DockWidgetArea) -> None:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_menus(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        act = file_menu.addAction("Open CSV…", self.on_open_csv)
        act.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        act = file_menu.addAction("Save cleaned CSV…", self.on_save_clean)
        act.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        file_menu.addAction("Save annotations…", self.on_save_annotations)
        file_menu.addAction("Load annotations…", self.on_load_annotations)
        file_menu.addAction("Export figure…", self.on_export_figure)
        file_menu.addSeparator()
        file_menu.addAction("New project…", self.on_new_project)
        file_menu.addAction("Open project…", self.on_open_project)
        file_menu.addAction("Save project", self.on_save_project)
        file_menu.addSeparator()
        act = file_menu.addAction("Quit", self.close)
        act.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        edit_menu = menubar.addMenu("&Edit")
        act = edit_menu.addAction("Undo", self.data_model.undo)
        act.setShortcut(QtGui.QKeySequence("U"))
        act = edit_menu.addAction("Redo", self.data_model.redo)
        act.setShortcut(QtGui.QKeySequence("R"))
        edit_menu.addAction("Preferences…", self.on_preferences)
        self.tools_menu = menubar.addMenu("&Tools")
        self.tools_menu.addAction("Filters…", self.on_filters)
        self.tools_menu.addAction("Rename columns…", self.on_rename_columns)
        self.tools_menu.addAction("Delete channels…", self.on_delete_channels)
        self.tools_menu.addAction("Duplicate channels…", self.on_duplicate_channels)
        self.tools_menu.addAction("Create derived channel…", self.on_derived_channel)
        self.tools_menu.addAction("Coordinate frames…", self.on_frames)
        self.tools_menu.addAction("3D mapping…", self.on_mapping)
        self.tools_menu.addAction("Derived frame transform…", self.on_frame_transform)
        self.tools_menu.addAction("Relative orientation…", self.on_relative_orientation)
        self.tools_menu.addAction("Calibration wizard…", self.on_calibration)
        self.tools_menu.addAction("Save transforms…", self.on_save_transforms)
        self.tools_menu.addAction("Load transforms…", self.on_load_transforms)
        self.tools_menu.addAction("Reload plugins", self._reload_plugins)
        self.tools_menu.addAction("Save recipe from history…", self.save_recipe)
        self.tools_menu.addAction("Apply recipe to trials…", self.apply_recipe_to_trials)
        self.tools_menu.addAction("Compare trials…", self.on_compare_trials)
        self.tools_menu.addAction("Guided wizard…", self.on_wizard)
        self._build_plugin_menu()
        self._build_3d_view_menu(menubar)
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("Shortcuts", self.on_shortcuts)
        help_menu.addAction("About", lambda: QtWidgets.QMessageBox.information(self, "About", "Time-Series Annotation Studio"))

    def _build_plugin_menu(self) -> None:
        # remove old plugin submenu if exists
        for act in getattr(self, "plugin_actions", []):
            self.tools_menu.removeAction(act)
        self.plugin_actions: List[QtGui.QAction] = []
        plugin_names = self.plugins.menu_entries()
        if not plugin_names:
            return
        self.tools_menu.addSeparator()
        for name in plugin_names:
            act = self.tools_menu.addAction(f"Plugin: {name}", lambda n=name: self.apply_plugin(n))
            self.plugin_actions.append(act)

    def _reload_plugins(self) -> None:
        self.plugins.load_plugins()
        self._build_plugin_menu()
        self.statusBar().showMessage("Plugins reloaded")

    def _build_3d_view_menu(self, menubar: QtWidgets.QMenuBar) -> None:
        """Build the 3D View menu with camera presets and zoom controls."""
        self.view_3d_menu = menubar.addMenu("3D &View")

        # View presets
        self.view_3d_menu.addAction("Reset View", lambda: self.plot3d.set_view_preset("reset"))
        self.view_3d_menu.addSeparator()
        self.view_3d_menu.addAction("Top View", lambda: self.plot3d.set_view_preset("top"))
        self.view_3d_menu.addAction("Front View", lambda: self.plot3d.set_view_preset("front"))
        self.view_3d_menu.addAction("Back View", lambda: self.plot3d.set_view_preset("back"))
        self.view_3d_menu.addAction("Side View (Right)", lambda: self.plot3d.set_view_preset("side"))
        self.view_3d_menu.addAction("Side View (Left)", lambda: self.plot3d.set_view_preset("left"))
        self.view_3d_menu.addAction("Isometric View", lambda: self.plot3d.set_view_preset("isometric"))

        # Zoom controls
        self.view_3d_menu.addSeparator()
        zoom_in_action = self.view_3d_menu.addAction("Zoom In", self.plot3d.zoom_in)
        zoom_in_action.setShortcut(QtGui.QKeySequence("Ctrl+="))
        zoom_out_action = self.view_3d_menu.addAction("Zoom Out", self.plot3d.zoom_out)
        zoom_out_action.setShortcut(QtGui.QKeySequence("Ctrl+-"))

    def _connect_signals(self) -> None:
        self.channel_manager.channelToggled.connect(self.update_channels)
        self.data_model.dataChanged.connect(self._on_data_changed)
        self.data_model.annotationsChanged.connect(self._on_annotations_changed)
        self.data_model.historyChanged.connect(self._on_history_changed)
        self.data_model.statusMessage.connect(self.statusBar().showMessage)     
        self.plot2d.widget.scene().sigMouseClicked.connect(self.on_plot_clicked)
        self.plot2d.set_selection_callback(self.on_region_dragged)
        self.plot2d.set_annotation_drag_callback(self.on_annotation_dragged)   
        self.filter_panel.applyRequested.connect(lambda: self.apply_filters_from_panel(preview=False))
        self.filter_panel.previewRequested.connect(lambda: self.apply_filters_from_panel(preview=True))
       
        self.style_panel.styleChanged.connect(self.on_channel_style_changed)       
        self.ann_table.itemSelectionChanged.connect(self.on_annotation_selected)
        self.ann_table.itemDoubleClicked.connect(self.on_annotation_edit) 
        self.suggestions.itemDoubleClicked.connect(self.on_accept_suggestion)
        self.play_action.triggered.connect(self.toggle_playback)
        self.stop_action.triggered.connect(self.stop_playback)
        self.prev_action.triggered.connect(lambda: self._nudge_time(-1))
        self.next_action.triggered.connect(lambda: self._nudge_time(1))
        self.overlay_action.toggled.connect(self.on_overlay_toggled)
        self.plot_style_combo.currentTextChanged.connect(self.on_plot_style_changed)
        self.season_period_spin.valueChanged.connect(self.on_season_period_changed)
        self.mode_group.idClicked.connect(self._on_mode_changed)
        self.show_3d_action.toggled.connect(self.toggle_3d_visibility)
        self.show_annotations_action.toggled.connect(self.toggle_annotations_visibility)
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        self.cursor_slider.valueChanged.connect(self._on_slider_changed)
        self.snap_index_chk.stateChanged.connect(self._on_snap_changed)
        self.snap_peak_chk.stateChanged.connect(self._on_snap_changed)
        self.play_timer.timeout.connect(self._advance_time)
        self.project_panel.trialSelected.connect(self._load_trial_from_project)
        self._build_plugin_menu()
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()
        # Global playback shortcut to avoid widget-level space handling conflicts
        self.play_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space), self)
        self.play_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.play_shortcut.activated.connect(self.toggle_playback)
        # Edit mode shortcut
        self.edit_mode_shortcut = QtGui.QShortcut(QtGui.QKeySequence("E"), self)
        self.edit_mode_shortcut.activated.connect(lambda: self.edit_mode_btn.setChecked(True))

    def on_open_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv)")
        if not path:
            return
        self.load_file(path)
        for t in self.project.trials:
            if t.path == path:
                t.status = "loaded"
        self.project_panel.refresh()

    def load_file(self, path: str) -> None:
        """Parse the CSV on a worker thread, then adopt it on the UI thread."""
        busy = QtWidgets.QProgressDialog(f"Loading {os.path.basename(path)}…", None, 0, 0, self)
        busy.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        busy.setMinimumDuration(300)

        def done(df: pd.DataFrame) -> None:
            busy.close()
            self.data_model.load_frame(df, path)
            self._after_load(path)

        def fail(exc: BaseException) -> None:
            busy.close()
            QtWidgets.QMessageBox.critical(
                self, "Load Error",
                f"Failed to load {os.path.basename(path)}:\n{type(exc).__name__}: {exc}"
            )

        run_in_background(DataModel.read_csv_frame, path, on_finished=done, on_error=fail)

    def _after_load(self, path: str) -> None:
        self.loaded_file_path = path
        self.filter_engine.set_sample_rate(self.data_model.sample_rate)
        groups = self.data_model.channel_groups()
        self.channel_manager.populate(self.data_model.time_columns, self.data_model.metadata_columns, groups)
        self.update_channels()
        self._update_episode_overlay()
        self.statusBar().showMessage(f"Loaded {os.path.basename(path)} | fs={self.data_model.sample_rate} Hz")
        self.project.update_status(path, "loaded", "Loaded into session")
        self.project_panel.refresh()
        self._run_suggestions()

    def on_save_clean(self) -> None:
        from project_manager import parse_trial_filename

        # Show export options dialog if annotations exist
        embed_annotations = True
        manual_indices = None

        if self.data_model.annotations:
            dlg = ExportCSVDialog(self.data_model.annotations, self)
            if not dlg.exec():
                return
            params = dlg.export_params()
            embed_annotations = params["embed_annotations"]
            manual_indices = params["manual_indices"]

        # Determine default save directory and filename
        base_dir = "/Users/avimehrotra/development/TDATA/MetricsData_Hybrid/EditedTrials"
        default_path = base_dir

        if self.loaded_file_path:
            parsed = parse_trial_filename(self.loaded_file_path)
            filename = os.path.basename(self.loaded_file_path)

            if parsed.parse_success and parsed.participant:
                # Create participant subfolder
                participant_dir = os.path.join(base_dir, parsed.participant)
                os.makedirs(participant_dir, exist_ok=True)
                default_path = os.path.join(participant_dir, filename)
            else:
                # Fallback: use base directory with original filename
                os.makedirs(base_dir, exist_ok=True)
                default_path = os.path.join(base_dir, filename)
        else:
            os.makedirs(base_dir, exist_ok=True)

        # Get save path with pre-populated default
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save cleaned CSV", default_path, "CSV files (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        # Save with annotation embedding
        self.data_model.save_clean(path, embed_annotations, manual_indices)
        self.project.update_status(path, "cleaned", "Cleaned CSV saved")
        self.project_panel.refresh()

    def on_save_annotations(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save annotations", "", "JSON files (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        self.data_model.save_annotations(path)

    def on_load_annotations(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load annotations", "", "JSON files (*.json)")
        if not path:
            return
        self.data_model.load_annotations(path)
        self._on_annotations_changed()

    def on_new_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "New project", "", "Project files (*.json)")
        if not path:
            return
        self.project.new_project(path)
        self.project_panel.refresh()

    def on_open_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", "", "Project files (*.json)")
        if not path:
            return
        self.project.load(path)
        self.project_panel.refresh()
        if self.project.trials:
            self.statusBar().showMessage("Project loaded. Double-click a trial to open.")

    def on_save_project(self) -> None:
        self.project.save()
        self.statusBar().showMessage("Project saved")

    def _load_trial_from_project(self, path: str) -> None:
        self.load_file(path)

    def on_preferences(self) -> None:
        dlg = PreferencesDialog(self.data_model.sample_rate, self)
        dlg.output_dir.setText(self.project.preferences.get("default_output_dir", ""))
        ui_state = load_ui_state()
        dlg.theme_combo.setCurrentText(ui_state.get("theme", "System"))
        if dlg.exec():
            vals = dlg.values()
            self.data_model.set_sample_rate(vals["fs"])
            self.filter_engine.set_sample_rate(vals["fs"])
            self.project.preferences["default_output_dir"] = vals["output_dir"]
            self.project.save()
            self.apply_theme_preference(vals["theme"])

    def apply_theme_preference(self, theme: str) -> None:
        """Apply and persist the theme, restyling plots to match."""
        app = QtWidgets.QApplication.instance()
        scheme = apply_theme(app, theme)
        self.plot2d.set_style(dark=(scheme == "dark"))
        self.plot2d.refresh_plots()
        ui_state = load_ui_state()
        ui_state["theme"] = theme
        save_ui_state(ui_state)

    def on_shortcuts(self) -> None:
        ShortcutDialog(self).exec()

    def on_wizard(self) -> None:
        GuidedWizard(self).exec()

    def on_frames(self) -> None:
        dlg = FrameManagerDialog(self.frames, self)
        if dlg.exec():
            self.frames = dlg.frames_data()
            self.plot3d.set_frames(self.frames)
        # Re-import episodes as annotation segments if available
        self._update_episode_overlay()

    def on_frame_transform(self) -> None:
        if self.data_model.df is None or not self.data_model.signal_columns:
            return
        cols = self.data_model.signal_columns
        src, ok = QtWidgets.QInputDialog.getItem(self, "Source heading", "Choose source", cols, editable=False)
        if not ok:
            return
        dst, ok = QtWidgets.QInputDialog.getItem(self, "Target heading", "Choose target", cols, editable=False)
        if not ok:
            return
        offset, _ = QtWidgets.QInputDialog.getDouble(self, "Offset", "Offset degrees", value=0.0, decimals=2)
        new_name, ok = QtWidgets.QInputDialog.getText(self, "New channel", "Name", text=f"{src}_vs_{dst}")
        if not ok or not new_name:
            return
        df = self.data_model.get_dataframe()
        if src not in df or dst not in df:
            return
        df[new_name] = ((df[src] - df[dst] - offset + 180) % 360) - 180
        if new_name not in self.data_model.signal_columns:
            self.data_model.signal_columns.append(new_name)
        self.data_model.apply_dataframe(df, "frame_transform", 0.0, df["normalized_time"].max(), {"source": src, "target": dst, "offset": offset})
        self._run_suggestions()

    def on_relative_orientation(self) -> None:
        """Open the relative orientation dialog to compute relative angles between segments."""
        if self.data_model.df is None or not self.data_model.signal_columns:
            QtWidgets.QMessageBox.information(
                self, "No Data",
                "Please load a CSV file first."
            )
            return

        dlg = RelativeOrientationDialog(
            columns=self.data_model.signal_columns,
            df=self.data_model.df,
            parent=self
        )

        if not dlg.exec():
            return

        params = dlg.params()
        df = self.data_model.get_dataframe()

        if params["mode"] == "heading":
            # Simple heading mode - compute relative heading
            src = params["source"]
            tgt = params["target"]
            offset = params["offset"]
            output_name = params["output"]

            if src not in df.columns or tgt not in df.columns:
                QtWidgets.QMessageBox.warning(
                    self, "Error",
                    "Source or target column not found in data."
                )
                return

            # Compute relative heading with proper wrapping
            df[output_name] = ((df[src] - df[tgt] - offset + 180) % 360) - 180

            if output_name not in self.data_model.signal_columns:
                self.data_model.signal_columns.append(output_name)

            self.data_model.apply_dataframe(
                df, "relative_heading", 0.0, df["normalized_time"].max(),
                {"source": src, "target": tgt, "offset": offset, "output": output_name}
            )
            self.statusBar().showMessage(f"Created relative heading channel: {output_name}")

        else:
            # Quaternion mode - compute relative rotation (yaw, pitch, roll)
            parent_cols = params["parent"]
            child_cols = params["child"]
            outputs = params["outputs"]

            # Validate columns exist
            for col in list(parent_cols.values()) + list(child_cols.values()):
                if col not in df.columns:
                    QtWidgets.QMessageBox.warning(
                        self, "Error",
                        f"Column '{col}' not found in data."
                    )
                    return

            # Compute relative rotation
            yaw, pitch, roll = self.filter_engine.relative_rotation(
                df,
                parent_cols["qw"], parent_cols["qx"], parent_cols["qy"], parent_cols["qz"],
                child_cols["qw"], child_cols["qx"], child_cols["qy"], child_cols["qz"]
            )

            created_channels = []
            if outputs["yaw"]:
                df[outputs["yaw"]] = yaw
                if outputs["yaw"] not in self.data_model.signal_columns:
                    self.data_model.signal_columns.append(outputs["yaw"])
                created_channels.append(outputs["yaw"])

            if outputs["pitch"]:
                df[outputs["pitch"]] = pitch
                if outputs["pitch"] not in self.data_model.signal_columns:
                    self.data_model.signal_columns.append(outputs["pitch"])
                created_channels.append(outputs["pitch"])

            if outputs["roll"]:
                df[outputs["roll"]] = roll
                if outputs["roll"] not in self.data_model.signal_columns:
                    self.data_model.signal_columns.append(outputs["roll"])
                created_channels.append(outputs["roll"])

            self.data_model.apply_dataframe(
                df, "relative_rotation", 0.0, df["normalized_time"].max(),
                {"parent": parent_cols, "child": child_cols, "outputs": outputs}
            )
            self.statusBar().showMessage(f"Created relative rotation channels: {', '.join(created_channels)}")

        self._run_suggestions()

    def on_calibration(self) -> None:
        if self.data_model.df is None:
            return
        # Get current selection from 2D plot (if any)
        selection = self.plot2d.get_selection()
        dlg = CalibrationWizard(
            self.data_model.signal_columns,
            df=self.data_model.df,
            current_selection=selection,
            parent=self
        )
        if not dlg.exec():
            return
        params = dlg.params()
        src = params["src"]
        ref = params["ref"]
        start = params["start"]
        end = params["end"]
        name = params["name"] or "calibration"
        df = self.data_model.take_time_slice(start, end)
        if df.empty or src not in df or ref not in df:
            QtWidgets.QMessageBox.warning(self, "Calibration failed", "Invalid channels or empty window.")
            return
        offset = float((df[src] - df[ref]).mean())
        self.frames[name] = {"parent": ref, "offset": offset}
        self.plot3d.set_frames(self.frames)
        QtWidgets.QMessageBox.information(self, "Calibration", f"Stored offset {offset:.2f} deg as frame '{name}'")

    def on_mapping(self) -> None:
        if self.data_model.df is None:
            return
        dlg = MappingDialog(list(self.data_model.df.columns), self)
        # Restore existing mapping if available
        if self.mapping:
            dlg.set_mapping(self.mapping)
        if dlg.exec():
            self.mapping = dlg.mapping()
            if self.mapping:
                self.show_3d_action.setChecked(True)
            self.statusBar().showMessage("3D mapping updated")
            self.plot3d.set_mappings(self.mapping)

    def on_save_transforms(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save transforms", "", "JSON files (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.frames, f, indent=2)
        self.statusBar().showMessage(f"Saved transforms to {path}")

    def on_load_transforms(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load transforms", "", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                frames = json.load(f)
            self.frames = frames
            self.plot3d.set_frames(self.frames)
            self.statusBar().showMessage(f"Loaded transforms from {path}")
        except json.JSONDecodeError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Load Error",
                f"Invalid JSON format in transforms file:\n{exc.msg} at line {exc.lineno}"
            )
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(
                self, "Load Error",
                f"Transforms file not found:\n{path}"
            )
        except PermissionError:
            QtWidgets.QMessageBox.warning(
                self, "Load Error",
                f"Permission denied reading transforms file:\n{path}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Load Error",
                f"Failed to load transforms:\n{type(exc).__name__}: {exc}"
            )

    def on_filters(self) -> None:
        if self.filter_dock:
            self.filter_dock.show()
            self.filter_dock.raise_()

    def on_rename_columns(self) -> None:
        """Open column rename dialog."""
        if self.data_model.df is None:
            QtWidgets.QMessageBox.information(
                self, "No Data", "Load a CSV file first."
            )
            return

        # Get columns excluding system columns
        all_cols = list(self.data_model.df.columns)
        exclude = {"normalized_time", "is_bad_segment"}
        cols_to_rename = [c for c in all_cols if c not in exclude]

        if not cols_to_rename:
            QtWidgets.QMessageBox.information(
                self, "No Columns", "No renameable columns found."
            )
            return

        dlg = ColumnRenameDialog(cols_to_rename, self)
        if not dlg.exec():
            return

        mappings = dlg.get_mappings()
        if not mappings:
            self.statusBar().showMessage("No columns renamed")
            return

        self.data_model.rename_channels(mappings)

        # Refresh UI
        self.channel_manager.populate(
            self.data_model.time_columns,
            self.data_model.metadata_columns,
            self.data_model.channel_groups()
        )
        self.plot2d.refresh_plots()
        self.statusBar().showMessage(f"Renamed {len(mappings)} column(s)")

    def on_delete_channels(self) -> None:
        """Open channel delete dialog."""
        if self.data_model.df is None:
            QtWidgets.QMessageBox.information(
                self, "No Data", "Load a CSV file first."
            )
            return

        # Get columns excluding system columns
        all_cols = list(self.data_model.df.columns)
        exclude = {"normalized_time", "is_bad_segment"}
        cols_to_delete = [c for c in all_cols if c not in exclude]

        if not cols_to_delete:
            QtWidgets.QMessageBox.information(
                self, "No Columns", "No deletable columns found."
            )
            return

        dlg = ChannelDeleteDialog(cols_to_delete, self)
        if not dlg.exec():
            return

        selected = dlg.get_selected_columns()
        if not selected:
            self.statusBar().showMessage("No channels deleted")
            return

        self.data_model.delete_channels(selected)

        # Refresh UI
        self.channel_manager.populate(
            self.data_model.time_columns,
            self.data_model.metadata_columns,
            self.data_model.channel_groups()
        )
        self.plot2d.refresh_plots()
        self.statusBar().showMessage(f"Deleted {len(selected)} channel(s)")

    def on_duplicate_channels(self) -> None:
        """Open channel duplicate dialog."""
        if self.data_model.df is None:
            QtWidgets.QMessageBox.information(
                self, "No Data", "Load a CSV file first."
            )
            return

        # Get columns excluding system columns
        all_cols = list(self.data_model.df.columns)
        exclude = {"normalized_time", "is_bad_segment"}
        cols_to_dup = [c for c in all_cols if c not in exclude]

        if not cols_to_dup:
            QtWidgets.QMessageBox.information(
                self, "No Columns", "No columns available to duplicate."
            )
            return

        dlg = ChannelDuplicateDialog(cols_to_dup, all_cols, self)
        if not dlg.exec():
            return

        mappings = dlg.get_mappings()
        if not mappings:
            self.statusBar().showMessage("No channels duplicated")
            return

        self.data_model.duplicate_channels(mappings)

        # Refresh UI
        self.channel_manager.populate(
            self.data_model.time_columns,
            self.data_model.metadata_columns,
            self.data_model.channel_groups()
        )
        self.plot2d.refresh_plots()
        self.statusBar().showMessage(f"Duplicated {len(mappings)} channel(s)")

    def on_derived_channel(self) -> None:
        """Open derived channel dialog."""
        if self.data_model.df is None:
            QtWidgets.QMessageBox.information(
                self, "No Data", "Load a CSV file first."
            )
            return

        # Get columns for expression building
        all_cols = list(self.data_model.df.columns)
        exclude = {"is_bad_segment"}
        available_cols = [c for c in all_cols if c not in exclude]

        if not available_cols:
            QtWidgets.QMessageBox.information(
                self, "No Columns", "No columns available for expressions."
            )
            return

        dlg = DerivedChannelDialog(available_cols, self.data_model.df, all_cols, self)
        if not dlg.exec():
            return

        params = dlg.get_params()
        if not params.get("name") or not params.get("expr"):
            self.statusBar().showMessage("No derived channel created")
            return

        # Validate expression for security
        is_valid, error = validate_plugin_expression(params["expr"], available_cols)
        if not is_valid:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Expression",
                f"Expression validation failed:\n{error}"
            )
            return

        try:
            self.data_model.create_derived_channel(params["name"], params["expr"])
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"Failed to create derived channel:\n{e}"
            )
            return

        # Refresh UI
        self.channel_manager.populate(
            self.data_model.time_columns,
            self.data_model.metadata_columns,
            self.data_model.channel_groups()
        )
        self.plot2d.refresh_plots()
        self.statusBar().showMessage(f"Created derived channel '{params['name']}'")

    def apply_filters_from_panel(self, preview: bool = False) -> None:
        if self.data_model.df is None:
            return
        chans = self.filter_panel.selected_channels()
        if not chans:
            self.statusBar().showMessage("Select at least one channel to filter")
            return
        try:
            params = self.filter_panel.parameters(preview=preview)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Invalid Parameters", str(e))
            return
        selection = None
        if params.pop("apply_selection") and all(self.selection):
            selection = tuple(sorted(self.selection))  # type: ignore
        filter_type = params.pop("filter")
        preview_flag = params.pop("preview", False)
        df_current = self.data_model.get_dataframe()

        # The filter math runs on a worker thread; the busy dialog keeps the
        # window modal-but-responsive. Cancel drops the result when it lands.
        busy = QtWidgets.QProgressDialog("Applying filter…", "Cancel", 0, 0, self)
        busy.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        busy.setMinimumDuration(300)

        def done(df_new: pd.DataFrame) -> None:
            busy.close()
            try:
                self._finish_filter(df_new, df_current, chans, filter_type, params,
                                    selection, preview_flag)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Filter Error",
                    f"Failed to apply filter:\n{type(exc).__name__}: {exc}"
                )

        def fail(exc: BaseException) -> None:
            busy.close()
            if isinstance(exc, ValueError):
                QtWidgets.QMessageBox.warning(
                    self, "Filter Error", f"Invalid filter parameters:\n{exc}"
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Filter Error",
                    f"Failed to apply filter:\n{type(exc).__name__}: {exc}"
                )

        job = run_in_background(
            self.filter_engine.apply,
            df_current, chans, filter_type, params,
            selection=selection,
            on_finished=done, on_error=fail,
        )
        busy.canceled.connect(job.cancel)

    def _finish_filter(
        self,
        df_new: pd.DataFrame,
        df_current: pd.DataFrame,
        chans: List[str],
        filter_type: str,
        params: Dict,
        selection: Optional[Tuple[float, float]],
        preview_flag: bool,
    ) -> None:
        """UI-thread continuation of apply_filters_from_panel."""
        if filter_type == "resample":
            self.data_model.set_sample_rate(params.get("target_fs", self.data_model.sample_rate))
        if preview_flag and chans:
            ch = chans[0]
            time = df_new["normalized_time"].to_numpy()
            orig_series = df_current[ch]
            orig_time = df_current["normalized_time"].to_numpy() if "normalized_time" in df_current else np.arange(len(orig_series))
            filt = df_new[ch].to_numpy()
            orig = orig_series.to_numpy()
            if len(time) != len(orig) or len(time) != len(filt):
                try:
                    # interpolate original onto new time base for preview
                    orig = np.interp(time, orig_time, orig_series)
                except Exception:
                    # last resort: truncate to common minimum length
                    n = min(len(time), len(orig), len(filt))
                    time = time[:n]
                    orig = orig[:n]
                    filt = filt[:n]
            prev_dlg = FilterPreviewDialog(time, orig, filt, ch, self)
            if not prev_dlg.exec():
                return

        start = selection[0] if selection else 0.0
        end = selection[1] if selection else df_new["normalized_time"].max()
        self.data_model.apply_dataframe(df_new, "filter", start, end, {"channels": chans, "filter_type": filter_type, **params})

    def save_recipe(self) -> None:
        if not self.data_model.history:
            QtWidgets.QMessageBox.information(self, "No history", "Perform some operations before saving a recipe.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save recipe", "", "JSON files (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        data = {"operations": [rec.model_dump() for rec in self.data_model.history]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.statusBar().showMessage(f"Recipe saved to {path}")

    def apply_recipe_to_trials(self) -> None:
        """Apply a recipe to trials with preview and custom output paths."""
        recipe_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open recipe", "", "JSON files (*.json)")
        if not recipe_path:
            return
        try:
            with open(recipe_path, "r", encoding="utf-8") as f:
                recipe = json.load(f)
        except json.JSONDecodeError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Recipe Error",
                f"Invalid JSON format in recipe file:\n{exc.msg} at line {exc.lineno}"
            )
            return
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(
                self, "Recipe Error",
                f"Recipe file not found:\n{recipe_path}"
            )
            return
        except PermissionError:
            QtWidgets.QMessageBox.warning(
                self, "Recipe Error",
                f"Permission denied reading recipe file:\n{recipe_path}"
            )
            return
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Recipe Error",
                f"Failed to load recipe:\n{type(exc).__name__}: {exc}"
            )
            return

        targets = self.project_panel.selected_trials()
        if not targets and self.data_model.df is not None:
            targets = ["__current__"]

        if not targets:
            QtWidgets.QMessageBox.information(
                self, "No Targets",
                "No trials selected and no data loaded."
            )
            return

        # Phase 1: Pre-process all trials without saving
        trial_results: List[Dict] = []
        progress = QtWidgets.QProgressDialog(
            "Processing trials...", "Cancel", 0, len(targets), self
        )
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        recipe_name = os.path.splitext(os.path.basename(recipe_path))[0]

        for i, trial_path in enumerate(targets):
            if progress.wasCanceled():
                self.statusBar().showMessage("Recipe application cancelled")
                return
            progress.setValue(i)
            progress.setLabelText(f"Processing {os.path.basename(trial_path) if trial_path != '__current__' else 'current session'}...")

            is_current = trial_path == "__current__"
            model = self.data_model if is_current else DataModel()

            if not is_current:
                try:
                    model.load_csv(trial_path)
                except FileNotFoundError:
                    QtWidgets.QMessageBox.warning(
                        self, "Load Error",
                        f"Trial file not found:\n{trial_path}"
                    )
                    continue
                except PermissionError:
                    QtWidgets.QMessageBox.warning(
                        self, "Load Error",
                        f"Permission denied reading trial:\n{trial_path}"
                    )
                    continue
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(
                        self, "Load Error",
                        f"Failed to load trial {os.path.basename(trial_path)}:\n{type(exc).__name__}: {exc}"
                    )
                    continue

            self.filter_engine.set_sample_rate(model.sample_rate)
            # For current session, store original for preview (can't reload from file)
            # For file trials, we'll reload on-demand to save memory
            original_df = model.get_dataframe().copy() if is_current else None
            df = model.get_dataframe().copy()  # Work on copy to avoid mutating original
            op_count = 0
            skipped_ops: List[str] = []

            for op in recipe.get("operations", []):
                desc = op.get("description")
                params = op.get("params", {})
                if desc == "filter":
                    chans = params.get("channels", model.signal_columns)
                    missing = [c for c in chans if c not in df.columns]
                    if missing:
                        filter_type = params.get("filter_type", params.get("filter", "unknown"))
                        skipped_ops.append(f"filter ({filter_type}): missing channels {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}")
                        continue
                    f_params = {k: v for k, v in params.items() if k != "channels"}
                    try:
                        df = self.filter_engine.apply(df, chans, f_params.get("filter_type", params.get("filter", "moving_average")), f_params)
                        op_count += 1
                    except Exception as e:
                        skipped_ops.append(f"filter: {type(e).__name__} - {str(e)[:50]}")

                elif desc == "rename":
                    mappings = params.get("mappings", {})
                    if not mappings:
                        continue
                    # Validate source columns exist
                    missing = [col for col in mappings.keys() if col not in df.columns]
                    if missing:
                        skipped_ops.append(f"rename: missing columns {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}")
                        continue
                    try:
                        df = df.rename(columns=mappings)
                        # Update model's column tracking
                        for old, new in mappings.items():
                            if old in model.signal_columns:
                                idx = model.signal_columns.index(old)
                                model.signal_columns[idx] = new
                            elif old in model.time_columns:
                                idx = model.time_columns.index(old)
                                model.time_columns[idx] = new
                            elif old in model.metadata_columns:
                                idx = model.metadata_columns.index(old)
                                model.metadata_columns[idx] = new
                        op_count += 1
                    except Exception as e:
                        skipped_ops.append(f"rename: {type(e).__name__}")

                elif desc == "delete_channels":
                    columns = params.get("columns", [])
                    if not columns:
                        continue
                    # Only delete columns that exist
                    existing = [c for c in columns if c in df.columns]
                    if not existing:
                        skipped_ops.append("delete_channels: no matching columns")
                        continue
                    try:
                        df = df.drop(columns=existing)
                        # Update model tracking lists
                        model.signal_columns = [c for c in model.signal_columns if c not in existing]
                        model.metadata_columns = [c for c in model.metadata_columns if c not in existing]
                        model.time_columns = [c for c in model.time_columns if c not in existing]
                        op_count += 1
                    except Exception as e:
                        skipped_ops.append(f"delete_channels: {type(e).__name__}")

                elif desc == "duplicate_channels":
                    mappings = params.get("mappings", {})
                    if not mappings:
                        continue
                    missing = [s for s in mappings.keys() if s not in df.columns]
                    if missing:
                        skipped_ops.append(f"duplicate_channels: missing {', '.join(missing[:3])}")
                        continue
                    try:
                        for source, new_name in mappings.items():
                            df[new_name] = df[source].copy()
                            if source in model.signal_columns:
                                model.signal_columns.append(new_name)
                            elif source in model.metadata_columns:
                                model.metadata_columns.append(new_name)
                        op_count += 1
                    except Exception as e:
                        skipped_ops.append(f"duplicate_channels: {type(e).__name__}")

                elif desc == "derived":
                    name = params.get("name")
                    expr = params.get("expr")
                    if not name or not expr:
                        continue
                    is_valid, error = validate_plugin_expression(expr, list(df.columns))
                    if not is_valid:
                        skipped_ops.append(f"derived ({name}): {error[:30]}")
                        continue
                    try:
                        df[name] = pd.eval(expr, local_dict=df.to_dict("series"))
                        if name not in model.signal_columns:
                            model.signal_columns.append(name)
                        op_count += 1
                    except Exception as e:
                        skipped_ops.append(f"derived ({name}): {type(e).__name__}")

                elif desc and desc.startswith("plugin:"):
                    plugin_name = desc.split(":", 1)[1]
                    try:
                        df, new_cols = self._apply_plugin_to_df(
                            plugin_name, df, model.signal_columns, show_warnings=False
                        )
                        for col in new_cols:
                            if col not in model.signal_columns:
                                model.signal_columns.append(col)
                        op_count += 1
                    except Exception as e:
                        skipped_ops.append(f"plugin ({plugin_name}): {type(e).__name__}")

            # Compute default output path
            if is_current:
                default_output = "(current session)"
            else:
                default_output = os.path.splitext(trial_path)[0] + f"_{recipe_name}.csv"

            result_entry = {
                "path": trial_path,
                "processed_df": df,
                "model": model,
                "signal_columns": list(model.signal_columns),
                "op_count": op_count,
                "skipped_ops": skipped_ops,
                "default_output": default_output,
            }
            # Only store original_df for current session (can't reload from file)
            if original_df is not None:
                result_entry["original_df"] = original_df
            trial_results.append(result_entry)

        progress.setValue(len(targets))

        if not trial_results:
            self.statusBar().showMessage("No trials processed")
            return

        # Phase 2: Show preview dialog
        preview_dialog = RecipePreviewDialog(recipe_name, trial_results, self)
        if not preview_dialog.exec():
            self.statusBar().showMessage("Recipe application cancelled")
            return

        # Phase 3: Save selected trials with custom paths
        selected = preview_dialog.get_selected_trials()
        summaries: List[str] = []

        for result in selected:
            trial_path = result["path"]
            out_path = result["output_path"]
            model = result["model"]
            df = result["processed_df"]
            op_count = result["op_count"]
            skipped_ops = result.get("skipped_ops", [])

            trial_name = os.path.basename(trial_path) if trial_path != "__current__" else "current session"

            if trial_path == "__current__":
                model.apply_dataframe(df, "recipe", 0.0, df["normalized_time"].max(), {"recipe": recipe_name})
                summary = f"Current session: {op_count} ops applied"
            else:
                model.set_dataframe(df)
                model.save_clean(out_path)
                summary = f"{trial_name} -> {os.path.basename(out_path)} ({op_count} ops)"
                self.project.update_status(trial_path, "cleaned", f"Recipe applied ({op_count} ops)")

            if skipped_ops:
                summary += f", {len(skipped_ops)} skipped"
            summaries.append(summary)

        if summaries:
            QtWidgets.QMessageBox.information(self, "Batch recipe summary", "\n".join(summaries))
        self.project_panel.refresh()
        self.statusBar().showMessage("Recipe applied")

    def _apply_plugin_to_df(
        self,
        name: str,
        df: pd.DataFrame,
        signal_columns: List[str],
        show_warnings: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Apply plugin operations to a DataFrame (for batch processing).

        Args:
            name: Plugin name
            df: DataFrame to modify
            signal_columns: List of signal column names
            show_warnings: Whether to show warning dialogs

        Returns:
            Tuple of (modified DataFrame, new signal columns added)
        """
        plugin = self.plugins.get_plugin(name)
        if not plugin:
            return df, []

        new_columns: List[str] = []
        ops = plugin.get("operations", [plugin])

        for op in ops:
            op_type = op.get("type", "")
            if op_type == "filter":
                channels = op.get("channels", signal_columns)
                ftype = op.get("filter", "moving_average")
                params = op.get("params", {})
                df = self.filter_engine.apply(df, channels, ftype, params)
            elif op_type == "derived":
                expr = op.get("expr")
                out = op.get("name", "derived")
                if expr:
                    # Security: Validate expression before execution
                    is_valid, error_msg = validate_plugin_expression(expr, list(df.columns))
                    if not is_valid:
                        if show_warnings:
                            QtWidgets.QMessageBox.warning(
                                self, "Plugin Security Error",
                                f"Expression blocked for security reasons:\n\n"
                                f"Expression: {expr[:100]}{'...' if len(expr) > 100 else ''}\n\n"
                                f"Reason: {error_msg}"
                            )
                        continue

                    try:
                        df[out] = pd.eval(expr, local_dict=df.to_dict("series"))
                        if out not in signal_columns:
                            new_columns.append(out)
                    except Exception as exc:
                        if show_warnings:
                            QtWidgets.QMessageBox.warning(self, "Plugin error", str(exc))

        return df, new_columns

    def apply_plugin(self, name: str) -> None:
        """Apply plugin to the current data model."""
        if self.data_model.df is None:
            return
        self.filter_engine.set_sample_rate(self.data_model.sample_rate)
        df = self.data_model.get_dataframe()

        df, new_columns = self._apply_plugin_to_df(
            name, df, self.data_model.signal_columns, show_warnings=True
        )

        # Add any new columns to signal_columns
        for col in new_columns:
            if col not in self.data_model.signal_columns:
                self.data_model.signal_columns.append(col)

        self.data_model.apply_dataframe(df, f"plugin:{name}", 0.0, df["normalized_time"].max(), {"plugin": name})

    def on_compare_trials(self) -> None:
        paths = [t.path for t in self.project.trials]
        if not paths:
            QtWidgets.QMessageBox.information(self, "No trials", "Add trials to the project first.")
            return
        channels = self.data_model.signal_columns if self.data_model.signal_columns else ["normalized_time"]
        dlg = CompareTrialsDialog(paths, channels, self)
        dlg.exec()

    def on_export_figure(self) -> None:
        if self.data_model.df is None:
            return
        dlg = ExportFigureDialog(self)
        if not dlg.exec():
            return
        params = dlg.export_params()
        fmt = params["format"].lower()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export figure", f"figure.{fmt}", f"*.{fmt}")
        if not path:
            return
        if not path.lower().endswith(f".{fmt}"):
            path += f".{fmt}"
        self._export_view(path, fmt, params)

    def _export_view(self, path: str, fmt: str, params: Dict) -> None:
        try:
            if fmt == "png":
                exporter = pg.exporters.ImageExporter(self.plot2d.widget.scene())
                exporter.parameters()["width"] = int(params.get("width_cm", 10) / 2.54 * params.get("dpi", 200))
                exporter.export(path)
            elif fmt == "svg":
                exporter = pg.exporters.SVGExporter(self.plot2d.widget.scene())
                exporter.export(path)
            elif fmt == "pdf":
                printer = QtGui.QPdfWriter(path)
                printer.setResolution(int(params.get("dpi", 200)))
                painter = QtGui.QPainter(printer)
                self.plot2d.widget.render(painter)
                painter.end()
            self.statusBar().showMessage(f"Exported figure to {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))

    def toggle_playback(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self.play_timer.start()
            self.statusBar().showMessage("Playback started")
        else:
            self.play_timer.stop()
            self.statusBar().showMessage("Playback paused")

    def stop_playback(self) -> None:
        self.playing = False
        self.play_timer.stop()
        self.set_time_cursor(0.0)
        self.statusBar().showMessage("Playback stopped")

    def toggle_3d_visibility(self, visible: bool) -> None:
        self.gl_container.setVisible(visible)
        if not visible:
            # widen 2D plot when hiding 3D
            self.splitter.setSizes([1, 0])
        else:
            # restore reasonable split
            self.splitter.setSizes([1200, 400])
        if visible and self.data_model.df is not None:
            self.plot3d.set_data(self.data_model.get_dataframe())
            self.plot3d.set_mappings(self.mapping)
            self.plot3d.set_frames(self.frames)

    def toggle_annotations_visibility(self, visible: bool) -> None:
        """Toggle visibility of annotation overlays on the 2D plot."""
        self.plot2d.set_annotations_visible(visible)
        self.statusBar().showMessage(f"Annotations {'shown' if visible else 'hidden'}")

    def _on_speed_changed(self, text: str) -> None:
        try:
            self.play_speed = float(text.rstrip("x"))
        except Exception:
            self.play_speed = 1.0

    def _advance_time(self) -> None:
        if self.data_model.df is None:
            return
        t_max = float(self.data_model.df["normalized_time"].max())
        cur = self.current_time
        delta = self.play_speed / self.data_model.sample_rate
        new_t = cur + delta
        if new_t >= t_max:
            new_t = 0.0
        self.set_time_cursor(new_t)

    def _on_slider_changed(self, value: int) -> None:
        if self.data_model.df is None:
            return
        t_max = float(self.data_model.df["normalized_time"].max())
        t = value / 1000.0 * t_max
        self.set_time_cursor(t)

    def _on_mode_changed(self, mode_id: int) -> None:
        """Handle interaction mode changes."""
        new_mode = InteractionMode(mode_id)

        # Warn if selection will be lost
        if self.selection[0] is not None and self.selection[1] is not None:
            if new_mode != self.interaction_mode:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Clear Selection?",
                    f"Switching to {new_mode.name.title()} mode will clear your current selection "
                    f"({self.selection[0]:.3f}s - {self.selection[1]:.3f}s).\n\n"
                    f"Continue?",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.Yes  # Default Yes since mode switch is intentional
                )
                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                    # Restore previous mode button state
                    self._restore_mode_button()
                    return

        self.interaction_mode = new_mode

        # Reset state when switching modes
        self.selection = (None, None)
        self.plot2d.clear_selection()

        # Update plot2d draggability BEFORE refreshing annotations
        self.plot2d.set_annotations_draggable(self.interaction_mode == InteractionMode.EDIT)

        # Force complete refresh of annotation regions to ensure clean state
        # This prevents stale event handlers from previous mode from interfering
        if self.data_model.df is not None:
            self.plot2d.update_annotations(self.data_model.annotations, self.data_model.deletions)
            # Respect visibility setting
            if not self.show_annotations_action.isChecked():
                self.plot2d.set_annotations_visible(False)

        # Status bar hint
        hints = {
            InteractionMode.TRIMMING: "Trim mode: Click start/end to select region, then D=delete, M=mark bad",
            InteractionMode.ANNOTATION: "Annotate mode: Click start/end to create annotation, Delete=remove selected",
            InteractionMode.EDIT: "Edit mode: Drag annotations to adjust, Shift+drag=start only, Ctrl+drag=end only",
        }
        self.statusBar().showMessage(hints[self.interaction_mode])

    def _restore_mode_button(self) -> None:
        """Restore mode button to current interaction mode (when user cancels mode switch)."""
        self.mode_group.blockSignals(True)
        if self.interaction_mode == InteractionMode.TRIMMING:
            self.trim_mode_btn.setChecked(True)
        elif self.interaction_mode == InteractionMode.ANNOTATION:
            self.annotate_mode_btn.setChecked(True)
        else:
            self.edit_mode_btn.setChecked(True)
        self.mode_group.blockSignals(False)

    def set_time_cursor(self, t: float) -> None:
        self.current_time = float(t)
        self.plot2d.set_time_cursor(t)
        if self.gl_container.isVisible():
            self.plot3d.update_time(t)
        if self.data_model.df is not None:
            self.cursor_slider.blockSignals(True)
            t_max = float(self.data_model.df["normalized_time"].max())
            val = int(1000 * t / t_max) if t_max > 0 else 0
            self.cursor_slider.setValue(val)
            self.cursor_slider.blockSignals(False)

    def on_plot_clicked(self, event: pg.GraphicsScene.mouseEvents) -> None:
        if not self.plot2d.plots:
            return
        pos = event.scenePos()
        vb = self.plot2d.plots[0].getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse_point = vb.mapSceneToView(pos)
        t = float(mouse_point.x())
        t = self._snap_time(t)

        if self.interaction_mode == InteractionMode.EDIT:
            # Edit mode: clicks select annotation at time (for deletion etc)
            self._select_annotation_at_time(t)
            return

        if self.interaction_mode == InteractionMode.ANNOTATION:
            # Annotation mode: two clicks create annotation
            if self.selection[0] is None:
                self.selection = (t, None)
                self.plot2d.set_selection(t, t + 0.05)
                self.statusBar().showMessage(f"Annotation start @ {t:.3f}s – click end point")
            else:
                self.selection = (self.selection[0], t)
                self._apply_selection_to_view()
                self._create_annotation_from_selection()
            return

        # Trimming mode: two clicks set selection for delete/mark
        if self.selection[0] is None:
            self.selection = (t, None)
        elif self.selection[1] is None:
            self.selection = (self.selection[0], t)
            self._apply_selection_to_view()
        else:
            self.selection = (t, None)
        self._draw_markers()

    def _snap_time(self, t: float) -> float:
        if self.data_model.df is None:
            return t
        time_col = self.data_model.df["normalized_time"].values
        if self.snap_to_index:
            idx = np.abs(time_col - t).argmin()
            t = float(time_col[idx])
        if self.snap_peak_chk.isChecked() and self.plot2d.channels:
            ch = self.plot2d.channels[0]
            vals = self.data_model.df[ch].values
            idx = np.abs(time_col - t).argmin()
            window = slice(max(0, idx - 3), min(len(vals), idx + 4))
            local = vals[window]
            if len(local) > 0:
                if abs(local.max() - vals[idx]) < abs(local.min() - vals[idx]):
                    idx_local = window.start + local.argmin()
                else:
                    idx_local = window.start + local.argmax()
                t = float(time_col[idx_local])
        return t

    def _apply_selection_to_view(self) -> None:
        if None not in self.selection:
            start, end = sorted(self.selection)  # type: ignore
            self.plot2d.set_selection(start, end)
            self.statusBar().showMessage(f"Selected {start:.3f}-{end:.3f} s")

    def _draw_markers(self) -> None:
        if self.selection[0] is None:
            self.plot2d.clear_selection()
        elif self.selection[1] is None:
            self.plot2d.set_selection(self.selection[0], self.selection[0] + 0.05)
        else:
            self._apply_selection_to_view()

    def on_region_dragged(self, start: float, end: float) -> None:
        self.selection = (start, end)
        self.statusBar().showMessage(f"Selected {start:.3f}-{end:.3f} s")

    def on_annotation_dragged(self, ann_id: int, start: float, end: float) -> None:
        # live update annotation boundaries without changing label/track/color
        self.data_model.update_annotation(ann_id, start, end, None, None, None)
        self._select_annotation_in_table(ann_id)
        self._on_annotations_changed()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        key = event.key()

        # Universal: Undo/Redo
        if key == QtCore.Qt.Key.Key_U:
            self.data_model.undo()
            return
        if key == QtCore.Qt.Key.Key_R:
            self.data_model.redo()
            return

        # Universal: Arrow keys for time nudge
        if key == QtCore.Qt.Key.Key_Left:
            self._nudge_time(-1)
            return
        if key == QtCore.Qt.Key.Key_Right:
            self._nudge_time(1)
            return

        # Mode-specific keys
        if self.interaction_mode == InteractionMode.TRIMMING:
            if key == QtCore.Qt.Key.Key_D:
                self.delete_selection()
            elif key == QtCore.Qt.Key.Key_M:
                self.mark_bad_selection()
            elif key == QtCore.Qt.Key.Key_A:
                self.annotate_selection()
            else:
                super().keyPressEvent(event)
        elif self.interaction_mode in (InteractionMode.ANNOTATION, InteractionMode.EDIT):
            if key in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
                self.delete_selected_annotation()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def _nudge_time(self, steps: int) -> None:
        if self.data_model.df is None:
            return
        dt = 1.0 / max(self.data_model.sample_rate, 1.0)
        if self.selection[0] is not None and self.selection[1] is not None:
            shift = steps * dt
            self.selection = (self.selection[0] + shift, self.selection[1] + shift)
            self._apply_selection_to_view()
        else:
            self.set_time_cursor(self.current_time + steps * dt)

    def _selection_values(self) -> Optional[Tuple[float, float]]:
        if None in self.selection:
            return None
        return tuple(sorted(self.selection))  # type: ignore

    def _annotation_at_time(self, t: float) -> Optional[AnnotationSegment]:
        matches = [a for a in self.data_model.annotations if a.start <= t <= a.end]
        if not matches:
            return None
        matches.sort(key=lambda a: (a.track != "episode", a.end - a.start))
        return matches[0]

    def _select_annotation_at_time(self, t: float) -> None:
        """Select annotation that contains the given time point (used in Edit mode)."""
        ann = self._annotation_at_time(t)
        if ann:
            self.selected_annotation_id = ann.id
            self.plot2d.highlight_annotation(ann.id)
            self._select_annotation_in_table(ann.id)
            self.statusBar().showMessage(f"Selected: {ann.label} ({ann.start:.2f}s – {ann.end:.2f}s)")
        else:
            # No annotation at this time - deselect
            self.selected_annotation_id = None
            self.plot2d.highlight_annotation(-1)
            self.statusBar().showMessage("Edit mode: Click on an annotation to select, then drag to adjust")

    def delete_selection(self) -> None:
        sel = self._selection_values()
        if not sel:
            return

        start, end = sel
        duration = end - start

        # Confirmation dialog for destructive operation
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Delete data segment from {start:.3f}s to {end:.3f}s ({duration:.3f}s)?\n\n"
            f"This will permanently remove {int(duration * self.data_model.sample_rate)} samples.\n"
            f"You can undo this operation with 'U' key.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No  # Default to No for safety
        )

        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self.data_model.delete_segment(*sel)
        self.selection = (None, None)
        self.plot2d.clear_selection()

    def mark_bad_selection(self) -> None:
        sel = self._selection_values()
        if not sel:
            return
        self.data_model.mark_bad(*sel)

    def annotate_selection(self) -> None:
        sel = self._selection_values()
        if not sel:
            return
        label, ok = QtWidgets.QInputDialog.getText(self, "Annotation label", "Label", text="blink")
        if not ok or not label:
            return
        track, _ = QtWidgets.QInputDialog.getText(self, "Track", "Track (e.g., eye, body)", text="default")
        color = "#4e79a7"
        self.data_model.annotate(*sel, label=label, track=track or "default", color=color)

    def _create_annotation_from_selection(self) -> None:
        """Create annotation when annotation mode is active."""
        sel = self._selection_values()
        if not sel:
            return
        default_label = self.last_annotation_label or "event"
        label, ok = QtWidgets.QInputDialog.getText(self, "Annotation label", "Label", text=default_label)
        if not ok or not label:
            return
        track, _ = QtWidgets.QInputDialog.getText(self, "Track", "Track (e.g., eye, body)", text="default")
        color = "#4e79a7"
        self.data_model.annotate(*sel, label=label, track=track or "default", color=color)
        self.last_annotation_label = label
        # prepare for next annotation
        self.selection = (None, None)
        self.plot2d.clear_selection()

    def _on_data_changed(self) -> None:
        df = self.data_model.get_dataframe()
        self.plot2d.set_data(df)
        self.plot3d.set_data(df)
        self.filter_panel.set_channels(self.data_model.signal_columns)
        self.style_panel.set_channels(self.data_model.signal_columns)
        if not df.empty:
            self.cursor_slider.setEnabled(True)
        self.autosave()
        self._run_suggestions()

    def _on_annotations_changed(self) -> None:
        self.ann_table.populate(self.data_model.annotations)
        self.plot2d.update_annotations(self.data_model.annotations, self.data_model.deletions)
        # Respect current visibility setting
        if not self.show_annotations_action.isChecked():
            self.plot2d.set_annotations_visible(False)
        if self.selected_annotation_id is not None:
            if any(a.id == self.selected_annotation_id for a in self.data_model.annotations):
                self.plot2d.highlight_annotation(self.selected_annotation_id)
            else:
                self.selected_annotation_id = None
                self.selection = (None, None)
                self.plot2d.clear_selection()
        self.autosave()

    def _on_history_changed(self) -> None:
        self.history_widget.clear()
        for record in self.data_model.history:
            self.history_widget.push(f"{record.description} [{record.start:.2f}-{record.end:.2f}] {record.params}")
        self.autosave()

    def update_channels(self) -> None:
        chans = self.channel_manager.get_checked_channels()
        self.plot2d.set_channels(chans)
        self.plot3d.set_active_channels({ch: "" for ch in chans})

    def on_annotation_selected(self) -> None:
        ann_id = self.ann_table.selected_annotation_id()
        if ann_id == -1:
            return
        self._select_annotation_in_table(ann_id, ensure_change=False)
        for ann in self.data_model.annotations:
            if ann.id == ann_id:
                start, end = sorted((ann.start, ann.end))
                self.selection = (start, end)
                self._apply_selection_to_view()
                self.set_time_cursor(start)
                self.plot2d.highlight_annotation(ann_id)
                self.plot2d.focus_on(start, end)
                break

    def _select_annotation_in_table(self, ann_id: int, ensure_change: bool = True) -> None:
        if ann_id == -1:
            return
        if ensure_change and ann_id == self.selected_annotation_id:
            return
        self.selected_annotation_id = ann_id
        self.ann_table.select_annotation(ann_id)

    def on_overlay_toggled(self, enabled: bool) -> None:
        self.plot2d.set_overlay_mode(enabled)
        self.update_channels()

    def on_plot_style_changed(self, label: str) -> None:
        style = self.plot_style_map.get(label, "line")
        self.plot2d.set_plot_style(style)
        seasonal = style == "seasonal"
        self.season_label.setVisible(seasonal)
        self.season_period_spin.setVisible(seasonal)

    def on_season_period_changed(self, period: float) -> None:
        self.plot2d.set_season_period(period)

    def on_channel_style_changed(self, channel: str, style: str) -> None:
        self.plot2d.set_channel_style(channel, style or None)

    def on_annotation_edit(self) -> None:
        ann_id = self.ann_table.selected_annotation_id()
        if ann_id == -1:
            return
        for ann in self.data_model.annotations:
            if ann.id == ann_id:
                dlg = QtWidgets.QDialog(self)
                dlg.setWindowTitle("Edit annotation")
                form = QtWidgets.QFormLayout(dlg)
                start = QtWidgets.QDoubleSpinBox()
                start.setRange(0, 1e6)
                start.setDecimals(3)
                start.setValue(ann.start)
                end = QtWidgets.QDoubleSpinBox()
                end.setRange(0, 1e6)
                end.setDecimals(3)
                end.setValue(ann.end)
                label = QtWidgets.QLineEdit(ann.label)
                track = QtWidgets.QLineEdit(ann.track)
                color_edit = QtWidgets.QLineEdit(ann.color)
                color_btn = QtWidgets.QPushButton("Pick…")
                def pick_color() -> None:
                    col = QtWidgets.QColorDialog.getColor(QtGui.QColor(color_edit.text()), self, "Choose color")
                    if col.isValid():
                        color_edit.setText(col.name())
                color_btn.clicked.connect(pick_color)
                color_row = QtWidgets.QHBoxLayout()
                color_row.addWidget(color_edit)
                color_row.addWidget(color_btn)
                # Episode index with auto checkbox
                episode_chk = QtWidgets.QCheckBox("Auto")
                episode_spin = QtWidgets.QSpinBox()
                episode_spin.setRange(1, 9999)
                if ann.episode_index is not None:
                    episode_spin.setValue(ann.episode_index)
                    episode_chk.setChecked(False)
                else:
                    episode_spin.setValue(1)
                    episode_chk.setChecked(True)
                episode_spin.setEnabled(not episode_chk.isChecked())
                episode_chk.toggled.connect(lambda checked: episode_spin.setEnabled(not checked))
                episode_row = QtWidgets.QHBoxLayout()
                episode_row.addWidget(episode_spin)
                episode_row.addWidget(episode_chk)
                form.addRow("Start", start)
                form.addRow("End", end)
                form.addRow("Label", label)
                form.addRow("Track", track)
                form.addRow("Color", color_row)
                form.addRow("Episode Index", episode_row)
                btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
                form.addRow(btns)
                btns.accepted.connect(dlg.accept)
                btns.rejected.connect(dlg.reject)
                if dlg.exec():
                    ep_idx = None if episode_chk.isChecked() else episode_spin.value()
                    self.data_model.update_annotation(ann_id, start.value(), end.value(), label.text(), track.text(), color_edit.text(), ep_idx)
                break

    def _show_annotation_menu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        edit_act = menu.addAction("Edit", self.on_annotation_edit)
        delete_act = menu.addAction("Delete", self.delete_annotation)
        jump_act = menu.addAction("Jump to segment", self.on_annotation_selected)
        menu.exec(self.ann_table.mapToGlobal(pos))

    def delete_annotation(self) -> None:
        ann_id = self.ann_table.selected_annotation_id()
        if ann_id == -1:
            return
        self.data_model.delete_annotation(ann_id)
        self.statusBar().showMessage("Annotation deleted (Ctrl+Z to undo)")

    def delete_selected_annotation(self) -> None:
        """Delete the currently selected annotation via keyboard (Delete/Backspace)."""
        # Check if we have a plot-selected annotation first
        if self.selected_annotation_id is not None:
            self.data_model.delete_annotation(self.selected_annotation_id)
            self.selected_annotation_id = None
            self._on_annotations_changed()
            self.statusBar().showMessage("Annotation deleted (Ctrl+Z to undo)")
            return

        # Fall back to table selection
        ann_id = self.ann_table.selected_annotation_id()
        if ann_id != -1:
            self.data_model.delete_annotation(ann_id)
            self._on_annotations_changed()
            self.statusBar().showMessage("Annotation deleted (Ctrl+Z to undo)")

    def _on_snap_changed(self) -> None:
        self.snap_to_index = self.snap_index_chk.isChecked()

    def _update_episode_overlay(self) -> None:
        if self.data_model.df is None:
            return
        df = self.data_model.df
        if "episode_index" not in df.columns or "episode_type" not in df.columns:
            return
        types = df["episode_type"].fillna("episode").astype(str)
        # ffill propagates episode index within episodes; NaN preserved for no-activity periods
        idxs = df["episode_index"].ffill()
        state_col = df["episode_state"] if "episode_state" in df.columns else None
        # remove prior episode annotations
        self.data_model.annotations = [a for a in self.data_model.annotations if not a.label.startswith("episode:")]
        next_id = self.data_model._id_counter
        max_id_seen = next_id
        for ep in idxs.dropna().unique():  # skip NaN (no-activity periods)
            ep_mask = idxs == ep
            start = df.loc[ep_mask, "normalized_time"].min()
            end = df.loc[ep_mask, "normalized_time"].max()
            label = types.loc[ep_mask].mode().iloc[0]
            lbl = f"episode:{label}"
            if state_col is not None:
                try:
                    state = state_col.loc[ep_mask].mode().iloc[0]
                    lbl = f"{lbl}:{state}"
                except Exception:
                    pass
            # default colors: inspection vs action
            color = "#8888ff"
            lbll = lbl.lower()
            if "action" in lbll:
                color = "#ffa500"
            elif "performing" in lbll:
                color = "#ffa500"
            elif "inspect" in lbll or "inspection" in lbll:
                color = "#6bd47a"
            try:
                ann_id = int(ep)
            except (TypeError, ValueError):
                ann_id = next_id
                next_id += 1
            max_id_seen = max(max_id_seen, ann_id + 1)
            self.data_model.annotations.append(
                AnnotationSegment(start=start, end=end, label=lbl, track="episode", color=color, id=ann_id)
            )
        self.data_model._id_counter = max_id_seen
        self._on_annotations_changed()

    def _run_suggestions(self) -> None:
        if self.data_model.df is None or not self.data_model.signal_columns:
            return
        df = self.data_model.df
        time = df["normalized_time"].to_numpy()
        ch = self.data_model.signal_columns[0]
        series = df[ch].to_numpy()
        deriv = np.abs(np.diff(series, prepend=series[0]))
        thr = np.nanmean(deriv) + 3 * np.nanstd(deriv)
        spike_mask = deriv > thr
        nan_mask = ~np.isfinite(series)
        segments: List[Tuple[float, float, str]] = []
        for mask, label in [(spike_mask, "spike"), (nan_mask, "nan")]:
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            start = idx[0]
            prev = idx[0]
            for i in idx[1:]:
                if i != prev + 1:
                    segments.append((time[start], time[prev], label))
                    start = i
                prev = i
            segments.append((time[start], time[prev], label))
        self.suggestion_segments = segments
        self.suggestions.clear()
        for s, e, label in segments:
            item = QtWidgets.QListWidgetItem(f"{label}: {s:.2f}-{e:.2f}s ({ch})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, (s, e, label))
            self.suggestions.addItem(item)

    def on_accept_suggestion(self, item: QtWidgets.QListWidgetItem) -> None:
        data = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not data:
            return
        s, e, label = data
        self.data_model.annotate(s, e, label=label, track="suggestion", color="#ffaa00")

    def autosave(self) -> None:
        try:
            state = {
                "schema_version": 2,  # Version 2 includes history and sample_rate
                "data": self.data_model.get_dataframe().to_dict(orient="list") if self.data_model.df is not None else None,
                "annotations": [ann.model_dump() for ann in self.data_model.annotations],
                "deletions": self.data_model.deletions,
                "history": [rec.model_dump() for rec in self.data_model.history],
                "sample_rate": self.data_model.sample_rate,
            }

            # Atomic write: write to temp file, then rename
            autosave_dir = os.path.dirname(self.autosave_path) or "."
            fd, temp_path = tempfile.mkstemp(suffix=".json", dir=autosave_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f)
                # Atomic rename (works on same filesystem)
                shutil.move(temp_path, self.autosave_path)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

        except PermissionError:
            self.statusBar().showMessage("Autosave failed: Permission denied", 5000)
        except OSError as e:
            self.statusBar().showMessage(f"Autosave failed: {e.strerror}", 5000)
        except Exception as e:
            print(f"Autosave error: {type(e).__name__}: {e}")
            self.statusBar().showMessage("Autosave failed", 5000)

    def prompt_restore_autosave(self) -> None:
        if not os.path.isfile(self.autosave_path):
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Restore previous session?",
            "A previous session autosave was found. Restore it?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            with open(self.autosave_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            data_dict = state.get("data")
            if data_dict:
                df = pd.DataFrame(data_dict)
                self.data_model.set_dataframe(df)

                # Restore annotations (robust deserialization)
                restored_annotations = []
                for a in state.get("annotations", []):
                    try:
                        restored_annotations.append(AnnotationSegment(**a))
                    except (TypeError, ValueError) as e:
                        print(f"Skipping malformed annotation: {e}")
                        continue
                self.data_model.annotations = restored_annotations

                self.data_model.deletions = [tuple(d) for d in state.get("deletions", [])]

                # Restore history (new in schema v2)
                restored_history = []
                for h in state.get("history", []):
                    try:
                        restored_history.append(OperationRecord(**h))
                    except (TypeError, ValueError) as e:
                        print(f"Skipping malformed history record: {e}")
                        continue
                self.data_model.history = restored_history

                # Restore sample rate (new in schema v2)
                if "sample_rate" in state:
                    try:
                        self.data_model.sample_rate = float(state["sample_rate"])
                    except (TypeError, ValueError):
                        pass

                groups = self.data_model.channel_groups()
                self.channel_manager.populate(self.data_model.time_columns, self.data_model.metadata_columns, groups)
                self.update_channels()
                self._on_annotations_changed()
                self.data_model.historyChanged.emit()
                self.statusBar().showMessage("Restored autosave session")
        except json.JSONDecodeError as e:
            QtWidgets.QMessageBox.warning(
                self, "Restore failed",
                f"Autosave file is corrupted:\n{e.msg} at line {e.lineno}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Restore failed",
                f"Could not restore autosave:\n{type(e).__name__}: {e}"
            )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.autosave()
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    apply_theme(app, load_ui_state().get("theme", "System"))
    pg.setConfigOptions(antialias=True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
