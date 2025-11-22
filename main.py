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
pip install PyQt6 pyqtgraph pandas numpy scipy

Run
---
python main.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401 - module provides exporters used dynamically
from PyQt6 import QtCore, QtGui, QtWidgets

from data_model import AnnotationSegment, DataModel
from dialogs import (
    AnnotationTable,
    ExportFigureDialog,
    FilterPanel,
    FilterPreviewDialog,
    FrameManagerDialog,
    MappingDialog,
    PreferencesDialog,
    ShortcutDialog,
    CompareTrialsDialog,
    CalibrationWizard,
)
from filter_engine import FilterEngine
from plot2d import PlotController2D
from plot3d import PlotController3D
from plugin_system import PluginManager
from project_manager import ProjectManager


class ChannelManagerWidget(QtWidgets.QWidget):
    """Panel listing time/metadata/signals with show/hide checkboxes."""

    channelToggled = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.time_list = QtWidgets.QListWidget()
        self.meta_list = QtWidgets.QListWidget()
        self.signal_container = QtWidgets.QScrollArea()
        self.signal_container.setWidgetResizable(True)
        self.signal_widget = QtWidgets.QWidget()
        self.signal_layout = QtWidgets.QVBoxLayout(self.signal_widget)
        self.signal_layout.setContentsMargins(0, 0, 0, 0)
        self.signal_container.setWidget(self.signal_widget)
        layout.addWidget(QtWidgets.QLabel("Time columns"))
        layout.addWidget(self.time_list)
        layout.addWidget(QtWidgets.QLabel("Metadata columns"))
        layout.addWidget(self.meta_list)
        layout.addWidget(QtWidgets.QLabel("Signals"))
        layout.addWidget(self.signal_container, 1)
        self.presets_combo = QtWidgets.QComboBox()
        self.presets_combo.setEditable(True)
        self.save_preset_btn = QtWidgets.QPushButton("Save preset")
        p_layout = QtWidgets.QHBoxLayout()
        p_layout.addWidget(self.presets_combo)
        p_layout.addWidget(self.save_preset_btn)
        layout.addLayout(p_layout)
        self.presets: Dict[str, List[str]] = {}
        self.save_preset_btn.clicked.connect(self.save_preset)
        self.presets_combo.currentIndexChanged.connect(self.apply_preset)

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

    def save_preset(self) -> None:
        name = self.presets_combo.currentText().strip()
        if not name:
            return
        self.presets[name] = self.get_checked_channels()
        if self.presets_combo.findText(name) == -1:
            self.presets_combo.addItem(name)

    def apply_preset(self) -> None:
        name = self.presets_combo.currentText()
        chans = self.presets.get(name, [])
        if not chans:
            return
        for i in range(self.signal_layout.count()):
            w = self.signal_layout.itemAt(i).widget()
            if isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(w.text() in chans)


class OperationHistoryWidget(QtWidgets.QListWidget):
    def push(self, text: str) -> None:
        self.addItem(text)
        self.scrollToBottom()


class ProjectPanel(QtWidgets.QWidget):
    trialSelected = QtCore.pyqtSignal(str)

    def __init__(self, project: ProjectManager, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.project = project
        layout = QtWidgets.QVBoxLayout(self)
        btns = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton("Add trial")
        self.save_btn = QtWidgets.QPushButton("Save project")
        btns.addWidget(self.add_btn)
        btns.addWidget(self.save_btn)
        layout.addLayout(btns)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Path", "Participant", "Condition", "Status", "Summary"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.add_btn.clicked.connect(self.add_trial)
        self.save_btn.clicked.connect(self.project.save)
        self.table.cellDoubleClicked.connect(self._emit_selection)

    def refresh(self) -> None:
        self.table.setRowCount(len(self.project.trials))
        for row, t in enumerate(self.project.trials):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(t.path))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(t.participant))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(t.condition))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(t.status))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(t.summary))

    def add_trial(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Add trial CSV", "", "CSV files (*.csv)")
        if not path:
            return
        participant, _ = QtWidgets.QInputDialog.getText(self, "Participant", "ID (optional)")
        condition, _ = QtWidgets.QInputDialog.getText(self, "Condition", "Condition (optional)")
        self.project.add_trial(path, participant, condition)
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
        self.plot2d = PlotController2D()
        self.plot3d = PlotController3D()
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(40)
        self.autosave_timer = QtCore.QTimer(self)
        self.autosave_timer.setInterval(120000)
        self.play_speed = 1.0
        self.snap_to_index = True
        self.playing = False
        self.selection: Tuple[Optional[float], Optional[float]] = (None, None)
        self.annotation_mode = False
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
        self.play_action = QtGui.QAction("Play/Pause", self)
        self.play_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space))
        self.toolbar.addAction(self.play_action)
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentText("1x")
        self.toolbar.addWidget(QtWidgets.QLabel("Speed"))
        self.toolbar.addWidget(self.speed_combo)
        self.overlay_action = QtGui.QAction("Overlay channels", self)
        self.overlay_action.setCheckable(True)
        self.overlay_action.setChecked(False)
        self.toolbar.addAction(self.overlay_action)
        self.annotation_mode_action = QtGui.QAction("Annotation mode", self)
        self.annotation_mode_action.setCheckable(True)
        self.annotation_mode_action.setToolTip("When enabled: click start/end points to create annotations quickly.")
        self.toolbar.addAction(self.annotation_mode_action)
        self.show_3d_action = QtGui.QAction("Show 3D", self)
        self.show_3d_action.setCheckable(True)
        self.show_3d_action.setChecked(False)
        self.toolbar.addAction(self.show_3d_action)
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
        self._add_dock("Channel Manager", self.channel_manager, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
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
        self.tools_menu.addAction("Coordinate frames…", self.on_frames)
        self.tools_menu.addAction("3D mapping…", self.on_mapping)
        self.tools_menu.addAction("Derived frame transform…", self.on_frame_transform)
        self.tools_menu.addAction("Calibration wizard…", self.on_calibration)
        self.tools_menu.addAction("Save transforms…", self.on_save_transforms)
        self.tools_menu.addAction("Load transforms…", self.on_load_transforms)
        self.tools_menu.addAction("Reload plugins", self._reload_plugins)
        self.tools_menu.addAction("Save recipe from history…", self.save_recipe)
        self.tools_menu.addAction("Apply recipe to trials…", self.apply_recipe_to_trials)
        self.tools_menu.addAction("Compare trials…", self.on_compare_trials)
        self.tools_menu.addAction("Guided wizard…", self.on_wizard)
        self._build_plugin_menu()
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
        self.ann_table.itemSelectionChanged.connect(self.on_annotation_selected)
        self.ann_table.itemDoubleClicked.connect(self.on_annotation_edit)
        self.suggestions.itemDoubleClicked.connect(self.on_accept_suggestion)
        self.play_action.triggered.connect(self.toggle_playback)
        self.overlay_action.toggled.connect(self.on_overlay_toggled)
        self.annotation_mode_action.toggled.connect(self.on_annotation_mode_toggled)
        self.show_3d_action.toggled.connect(self.toggle_3d_visibility)
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        self.cursor_slider.valueChanged.connect(self._on_slider_changed)
        self.snap_index_chk.stateChanged.connect(self._on_snap_changed)
        self.snap_peak_chk.stateChanged.connect(self._on_snap_changed)
        self.play_timer.timeout.connect(self._advance_time)
        self.project_panel.trialSelected.connect(self._load_trial_from_project)
        self._build_plugin_menu()
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()

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
        self.data_model.load_csv(path)
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
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save cleaned CSV", "", "CSV files (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        self.data_model.save_clean(path)
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
        if dlg.exec():
            vals = dlg.values()
            self.data_model.set_sample_rate(vals["fs"])
            self.filter_engine.set_sample_rate(vals["fs"])
            self.project.preferences["default_output_dir"] = vals["output_dir"]
            self.project.save()

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

    def on_calibration(self) -> None:
        if self.data_model.df is None:
            return
        dlg = CalibrationWizard(self.data_model.signal_columns, self)
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
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load error", str(exc))

    def on_filters(self) -> None:
        if self.filter_dock:
            self.filter_dock.show()
            self.filter_dock.raise_()

    def apply_filters_from_panel(self, preview: bool = False) -> None:
        if self.data_model.df is None:
            return
        chans = self.filter_panel.selected_channels()
        if not chans:
            self.statusBar().showMessage("Select at least one channel to filter")
            return
        params = self.filter_panel.parameters(preview=preview)
        selection = None
        if params.pop("apply_selection") and all(self.selection):
            selection = tuple(sorted(self.selection))  # type: ignore
        filter_type = params.pop("filter")
        preview_flag = params.pop("preview", False)
        df_current = self.data_model.get_dataframe()
        df_new = self.filter_engine.apply(
            df_current, chans, filter_type, params, selection=selection
        )
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
        data = {"operations": [rec.__dict__ for rec in self.data_model.history]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.statusBar().showMessage(f"Recipe saved to {path}")

    def apply_recipe_to_trials(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open recipe", "", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                recipe = json.load(f)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Recipe error", str(exc))
            return
        targets = self.project_panel.selected_trials()
        if not targets and self.data_model.df is not None:
            # apply to current only
            targets = ["__current__"]
        summaries: List[str] = []
        for trial_path in targets:
            model = self.data_model if trial_path == "__current__" else DataModel()
            if trial_path != "__current__":
                try:
                    model.load_csv(trial_path)
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Load error", f"{trial_path}: {exc}")
                    continue
            self.filter_engine.set_sample_rate(model.sample_rate)
            df = model.get_dataframe()
            op_count = 0
            for op in recipe.get("operations", []):
                desc = op.get("description")
                params = op.get("params", {})
                if desc == "filter":
                    chans = params.get("channels", model.signal_columns)
                    f_params = {k: v for k, v in params.items() if k != "channels"}
                    df = self.filter_engine.apply(df, chans, f_params.get("filter_type", params.get("filter", "moving_average")), f_params)
                    op_count += 1
                elif desc and desc.startswith("plugin:"):
                    plugin_name = desc.split(":", 1)[1]
                    self.apply_plugin(plugin_name)
                    op_count += 1
            if trial_path == "__current__":
                model.apply_dataframe(df, "recipe", 0.0, df["normalized_time"].max(), {"recipe": os.path.basename(path)})
                summaries.append(f"Current session: {op_count} ops")
            else:
                model.set_dataframe(df)
                out_path = os.path.splitext(trial_path)[0] + "_recipe.csv"
                model.save_clean(out_path)
                summaries.append(f"{os.path.basename(trial_path)} → {os.path.basename(out_path)} ({op_count} ops)")
                self.project.update_status(trial_path, "cleaned", f"Recipe applied ({op_count} ops)")
        if summaries:
            QtWidgets.QMessageBox.information(self, "Batch recipe summary", "\n".join(summaries))
        self.project_panel.refresh()
        self.statusBar().showMessage("Recipe applied")

    def apply_plugin(self, name: str) -> None:
        plugin = self.plugins.get_plugin(name)
        if not plugin or self.data_model.df is None:
            return
        self.filter_engine.set_sample_rate(self.data_model.sample_rate)
        df = self.data_model.get_dataframe()
        ops = plugin.get("operations", [plugin])
        for op in ops:
            op_type = op.get("type", "")
            if op_type == "filter":
                channels = op.get("channels", self.data_model.signal_columns)
                ftype = op.get("filter", "moving_average")
                params = op.get("params", {})
                df = self.filter_engine.apply(df, channels, ftype, params)
            elif op_type == "derived":
                expr = op.get("expr")
                out = op.get("name", "derived")
                if expr:
                    try:
                        df[out] = pd.eval(expr, local_dict=df.to_dict("series"))
                        if out not in self.data_model.signal_columns:
                            self.data_model.signal_columns.append(out)
                    except Exception as exc:
                        QtWidgets.QMessageBox.warning(self, "Plugin error", str(exc))
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
        else:
            self.play_timer.stop()

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

    def _on_speed_changed(self, text: str) -> None:
        try:
            self.play_speed = float(text.rstrip("x"))
        except Exception:
            self.play_speed = 1.0

    def _advance_time(self) -> None:
        if self.data_model.df is None:
            return
        t_max = float(self.data_model.df["normalized_time"].max())
        cur = self.cursor_slider.value() / 1000.0 * t_max
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

    def on_annotation_mode_toggled(self, enabled: bool) -> None:
        self.annotation_mode = enabled
        # reset any in-progress selection to avoid accidental deletes
        self.selection = (None, None)
        self.plot2d.clear_selection()
        if enabled:
            self.statusBar().showMessage("Annotation mode ON: click start then end to create an annotation.")
        else:
            self.statusBar().showMessage("Annotation mode OFF")

    def set_time_cursor(self, t: float) -> None:
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
        if self.annotation_mode:
            if self.selection[0] is None:
                self.selection = (t, None)
                self.plot2d.set_selection(t, t + 0.05)
                self.statusBar().showMessage(f"Annotation start @ {t:.3f}s")
            else:
                self.selection = (self.selection[0], t)
                self._apply_selection_to_view()
                self._create_annotation_from_selection()
            return
        # normal selection handling
        if self.selection[0] is None:
            self.selection = (t, None)
        elif self.selection[1] is None:
            self.selection = (self.selection[0], t)
            self._apply_selection_to_view()
        else:
            self.selection = (t, None)
        self._draw_markers()
        ann = self._annotation_at_time(t)
        if ann:
            self._select_annotation_in_table(ann.id)

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
        if key == QtCore.Qt.Key.Key_D:
            self.delete_selection()
        elif key == QtCore.Qt.Key.Key_M:
            self.mark_bad_selection()
        elif key == QtCore.Qt.Key.Key_A:
            self.annotate_selection()
        elif key == QtCore.Qt.Key.Key_U:
            self.data_model.undo()
        elif key == QtCore.Qt.Key.Key_R:
            self.data_model.redo()
        elif key == QtCore.Qt.Key.Key_Left:
            self._nudge_time(-1)
        elif key == QtCore.Qt.Key.Key_Right:
            self._nudge_time(1)
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
            current = self.cursor_slider.value() / 1000.0 * float(self.data_model.df["normalized_time"].max())
            self.set_time_cursor(current + steps * dt)

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

    def delete_selection(self) -> None:
        sel = self._selection_values()
        if not sel:
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
        if not df.empty:
            self.cursor_slider.setEnabled(True)
        self.autosave()
        self._run_suggestions()

    def _on_annotations_changed(self) -> None:
        self.ann_table.populate(self.data_model.annotations)
        self.plot2d.update_annotations(self.data_model.annotations, self.data_model.deletions)
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
                form.addRow("Start", start)
                form.addRow("End", end)
                form.addRow("Label", label)
                form.addRow("Track", track)
                form.addRow("Color", color_row)
                btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
                form.addRow(btns)
                btns.accepted.connect(dlg.accept)
                btns.rejected.connect(dlg.reject)
                if dlg.exec():
                    self.data_model.update_annotation(ann_id, start.value(), end.value(), label.text(), track.text(), color_edit.text())
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

    def _on_snap_changed(self) -> None:
        self.snap_to_index = self.snap_index_chk.isChecked()

    def _update_episode_overlay(self) -> None:
        if self.data_model.df is None:
            return
        df = self.data_model.df
        if "episode_index" not in df.columns or "episode_type" not in df.columns:
            return
        types = df["episode_type"].fillna("episode").astype(str)
        idxs = df["episode_index"].ffill().astype(int)
        state_col = df["episode_state"] if "episode_state" in df.columns else None
        # remove prior episode annotations
        self.data_model.annotations = [a for a in self.data_model.annotations if not a.label.startswith("episode:")]
        next_id = self.data_model._id_counter
        max_id_seen = next_id
        for ep in idxs.unique():
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
                "data": self.data_model.get_dataframe().to_dict(orient="list") if self.data_model.df is not None else None,
                "annotations": [ann.__dict__ for ann in self.data_model.annotations],
                "deletions": self.data_model.deletions,
            }
            with open(self.autosave_path, "w", encoding="utf-8") as f:
                json.dump(state, f)
        except Exception:
            pass

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
                self.data_model.annotations = [AnnotationSegment(**a) for a in state.get("annotations", [])]
                self.data_model.deletions = [tuple(d) for d in state.get("deletions", [])]
                groups = self.data_model.channel_groups()
                self.channel_manager.populate(self.data_model.time_columns, self.data_model.metadata_columns, groups)
                self.update_channels()
                self._on_annotations_changed()
                self.statusBar().showMessage("Restored autosave session")
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Restore failed", "Could not restore autosave.")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.autosave()
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
