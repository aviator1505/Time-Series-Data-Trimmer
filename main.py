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
    FilterDialog,
    FrameManagerDialog,
    MappingDialog,
    PreferencesDialog,
    ShortcutDialog,
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
                cb.setChecked(grp != "Other" and len(self.get_checked_channels()) < 6)
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
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Path", "Participant", "Condition", "Status"])
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
        self._build_ui()
        self._connect_signals()
        self.plugins.load_plugins()
        self.restore_autosave()

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
        self.ann_table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.ann_table.customContextMenuRequested.connect(self._show_annotation_menu)
        self._add_dock("Channel Manager", self.channel_manager, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
        self._add_dock("Annotations", self.ann_table, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self._add_dock("Operation History", self.history_widget, QtCore.Qt.DockWidgetArea.BottomDockWidgetArea)
        self._add_dock("Project", self.project_panel, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
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
        self.tools_menu.addAction("Reload plugins", self._reload_plugins)
        self.tools_menu.addAction("Save recipe from history…", self.save_recipe)
        self.tools_menu.addAction("Apply recipe to trials…", self.apply_recipe_to_trials)
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
        self.ann_table.itemSelectionChanged.connect(self.on_annotation_selected)
        self.ann_table.itemDoubleClicked.connect(self.on_annotation_edit)
        self.play_action.triggered.connect(self.toggle_playback)
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

    def on_save_clean(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save cleaned CSV", "", "CSV files (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        self.data_model.save_clean(path)

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

    def on_mapping(self) -> None:
        if self.data_model.df is None:
            return
        dlg = MappingDialog(list(self.data_model.df.columns), self)
        if dlg.exec():
            self.mapping = dlg.mapping()
            if self.mapping:
                self.show_3d_action.setChecked(True)
            self.statusBar().showMessage("3D mapping updated")

    def on_filters(self) -> None:
        if self.data_model.df is None:
            return
        dlg = FilterDialog(self.data_model.signal_columns, self)
        if not dlg.exec():
            return
        chans = dlg.selected_channels()
        params = dlg.parameters()
        selection = None
        if params.pop("apply_selection") and all(self.selection):
            selection = tuple(sorted(self.selection))  # type: ignore
        filter_type = params.pop("filter")
        df_new = self.filter_engine.apply(
            self.data_model.get_dataframe(), chans, filter_type, params, selection=selection
        )
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
            for op in recipe.get("operations", []):
                desc = op.get("description")
                params = op.get("params", {})
                if desc == "filter":
                    chans = params.get("channels", model.signal_columns)
                    f_params = {k: v for k, v in params.items() if k != "channels"}
                    df = self.filter_engine.apply(df, chans, f_params.get("filter_type", params.get("filter", "moving_average")), f_params)
                elif desc and desc.startswith("plugin:"):
                    plugin_name = desc.split(":", 1)[1]
                    self.apply_plugin(plugin_name)
            if trial_path == "__current__":
                model.apply_dataframe(df, "recipe", 0.0, df["normalized_time"].max(), {"recipe": os.path.basename(path)})
            else:
                model.set_dataframe(df)
                out_path = os.path.splitext(trial_path)[0] + "_recipe.csv"
                model.save_clean(out_path)
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

    def _on_data_changed(self) -> None:
        df = self.data_model.get_dataframe()
        self.plot2d.set_data(df)
        self.plot3d.set_data(df)
        if not df.empty:
            self.cursor_slider.setEnabled(True)
        self.autosave()

    def _on_annotations_changed(self) -> None:
        self.ann_table.populate(self.data_model.annotations)
        self.plot2d.update_annotations(self.data_model.annotations, self.data_model.deletions)
        self.autosave()

    def _on_history_changed(self) -> None:
        self.history_widget.clear()
        for record in self.data_model.history:
            self.history_widget.push(f"{record.description} [{record.start:.2f}-{record.end:.2f}] {record.params}")
        self.autosave()

    def update_channels(self) -> None:
        chans = self.channel_manager.get_checked_channels()
        self.plot2d.set_channels(chans)

    def on_annotation_selected(self) -> None:
        ann_id = self.ann_table.selected_annotation_id()
        if ann_id == -1:
            return
        for ann in self.data_model.annotations:
            if ann.id == ann_id:
                self.selection = (ann.start, ann.end)
                self._apply_selection_to_view()
                self.set_time_cursor(ann.start)
                break

    def on_annotation_edit(self) -> None:
        ann_id = self.ann_table.selected_annotation_id()
        if ann_id == -1:
            return
        for ann in self.data_model.annotations:
            if ann.id == ann_id:
                new_label, ok = QtWidgets.QInputDialog.getText(self, "Edit annotation", "Label", text=ann.label)
                if ok and new_label:
                    ann.label = new_label
                new_track, ok = QtWidgets.QInputDialog.getText(self, "Track", "Track", text=ann.track)
                if ok and new_track:
                    ann.track = new_track
                self._on_annotations_changed()
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
        self.data_model.annotations = [a for a in self.data_model.annotations if a.id != ann_id]
        self._on_annotations_changed()

    def _on_snap_changed(self) -> None:
        self.snap_to_index = self.snap_index_chk.isChecked()

    def _update_episode_overlay(self) -> None:
        if self.data_model.df is None:
            return
        df = self.data_model.df
        if "episode_index" not in df.columns or "episode_type" not in df.columns:
            return
        types = df["episode_type"].fillna("episode").astype(str)
        idxs = df["episode_index"].fillna(method="ffill").astype(int)
        for ep in idxs.unique():
            ep_mask = idxs == ep
            start = df.loc[ep_mask, "normalized_time"].min()
            end = df.loc[ep_mask, "normalized_time"].max()
            label = types.loc[ep_mask].mode().iloc[0]
            self.data_model.annotations.append(AnnotationSegment(start=start, end=end, label=f"episode:{label}"))
        self._on_annotations_changed()

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

    def restore_autosave(self) -> None:
        if not os.path.isfile(self.autosave_path):
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
            pass

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
