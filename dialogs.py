"""UI dialogs used across the application."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from data_model import AnnotationSegment
from filter_engine import available_filters


class FilterDialog(QtWidgets.QDialog):
    def __init__(self, channels: List[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Apply Filter")
        self.resize(460, 420)
        self.preview_requested = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Select channels:"))
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        for ch in channels:
            item = QtWidgets.QListWidgetItem(ch)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)
        layout.addWidget(QtWidgets.QLabel("Preset:"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems([
            "None",
            "Gaze smoothing (savgol 11/2)",
            "Head LPF 6 Hz",
            "Resample 60 Hz",
            "Normalize z-score",
        ])
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        layout.addWidget(self.preset_combo)
        # Filter type and parameters
        form = QtWidgets.QFormLayout()
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(available_filters())
        form.addRow("Filter type", self.filter_combo)
        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(3, 1001)
        self.window_spin.setValue(11)
        form.addRow("Window", self.window_spin)
        self.poly_spin = QtWidgets.QSpinBox()
        self.poly_spin.setRange(1, 5)
        self.poly_spin.setValue(2)
        form.addRow("Poly order", self.poly_spin)
        self.cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.cutoff_spin.setRange(0.1, 60.0)
        self.cutoff_spin.setDecimals(2)
        self.cutoff_spin.setValue(6.0)
        form.addRow("Low cutoff (Hz)", self.cutoff_spin)
        self.high_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.high_cutoff_spin.setRange(0.1, 60.0)
        self.high_cutoff_spin.setDecimals(2)
        self.high_cutoff_spin.setValue(10.0)
        form.addRow("High cutoff (Hz)", self.high_cutoff_spin)
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 6)
        self.order_spin.setValue(2)
        form.addRow("Order", self.order_spin)
        self.target_fs_spin = QtWidgets.QDoubleSpinBox()
        self.target_fs_spin.setRange(1.0, 1000.0)
        self.target_fs_spin.setDecimals(2)
        self.target_fs_spin.setValue(60.0)
        form.addRow("Target fs (Hz)", self.target_fs_spin)
        layout.addLayout(form)
        self.apply_selection_chk = QtWidgets.QCheckBox("Only apply to current selection")
        layout.addWidget(self.apply_selection_chk)
        btns = QtWidgets.QDialogButtonBox()
        self.preview_btn = btns.addButton("Preview", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        btns.addButton(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        btns.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self.preview_btn.clicked.connect(self._preview)
        layout.addWidget(btns)

    def selected_channels(self) -> List[str]:
        chans: List[str] = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                chans.append(item.text())
        return chans

    def parameters(self) -> Dict:
        return {
            "preset": self.preset_combo.currentText(),
            "filter": self.filter_combo.currentText(),
            "window": self.window_spin.value(),
            "polyorder": self.poly_spin.value(),
            "cutoff": self.cutoff_spin.value(),
            "high_cut": self.high_cutoff_spin.value(),
            "order": self.order_spin.value(),
            "target_fs": self.target_fs_spin.value(),
            "apply_selection": self.apply_selection_chk.isChecked(),
            "preview": self.preview_requested,
        }

    def _preview(self) -> None:
        self.preview_requested = True
        self.accept()

    def _apply_preset(self) -> None:
        preset = self.preset_combo.currentText()
        if preset == "Gaze smoothing (savgol 11/2)":
            self.filter_combo.setCurrentText("savgol")
            self.window_spin.setValue(11)
            self.poly_spin.setValue(2)
        elif preset == "Head LPF 6 Hz":
            self.filter_combo.setCurrentText("butter_lowpass")
            self.cutoff_spin.setValue(6.0)
            self.order_spin.setValue(2)
        elif preset == "Resample 60 Hz":
            self.filter_combo.setCurrentText("resample")
            self.target_fs_spin.setValue(60.0)
        elif preset == "Normalize z-score":
            self.filter_combo.setCurrentText("normalize_zscore")


class FilterPanel(QtWidgets.QWidget):
    applyRequested = QtCore.pyqtSignal()
    previewRequested = QtCore.pyqtSignal()

    def __init__(self, channels: List[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Filters"))
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        layout.addWidget(self.list_widget)
        btn_row = QtWidgets.QHBoxLayout()
        self.select_all_btn = QtWidgets.QPushButton("Select all")
        self.unselect_all_btn = QtWidgets.QPushButton("Unselect all")
        btn_row.addWidget(self.select_all_btn)
        btn_row.addWidget(self.unselect_all_btn)
        layout.addLayout(btn_row)
        self.set_channels(channels)
        layout.addWidget(QtWidgets.QLabel("Preset"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(
            [
                "None",
                "Gaze smoothing (savgol 11/2)",
                "Head LPF 6 Hz",
                "Resample 60 Hz",
                "Normalize z-score",
            ]
        )
        layout.addWidget(self.preset_combo)
        form = QtWidgets.QFormLayout()
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(available_filters())
        form.addRow("Filter type", self.filter_combo)
        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(3, 1001)
        self.window_spin.setValue(11)
        form.addRow("Window", self.window_spin)
        self.poly_spin = QtWidgets.QSpinBox()
        self.poly_spin.setRange(1, 5)
        self.poly_spin.setValue(2)
        form.addRow("Poly order", self.poly_spin)
        self.cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.cutoff_spin.setRange(0.1, 60.0)
        self.cutoff_spin.setDecimals(2)
        self.cutoff_spin.setValue(6.0)
        form.addRow("Low cutoff (Hz)", self.cutoff_spin)
        self.high_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.high_cutoff_spin.setRange(0.1, 60.0)
        self.high_cutoff_spin.setDecimals(2)
        self.high_cutoff_spin.setValue(10.0)
        form.addRow("High cutoff (Hz)", self.high_cutoff_spin)
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 6)
        self.order_spin.setValue(2)
        form.addRow("Order", self.order_spin)
        self.target_fs_spin = QtWidgets.QDoubleSpinBox()
        self.target_fs_spin.setRange(1.0, 1000.0)
        self.target_fs_spin.setDecimals(2)
        self.target_fs_spin.setValue(60.0)
        form.addRow("Target fs (Hz)", self.target_fs_spin)
        layout.addLayout(form)
        self.apply_selection_chk = QtWidgets.QCheckBox("Only apply to current selection")
        layout.addWidget(self.apply_selection_chk)
        btns = QtWidgets.QHBoxLayout()
        self.preview_btn = QtWidgets.QPushButton("Preview")
        self.apply_btn = QtWidgets.QPushButton("Apply")
        btns.addWidget(self.preview_btn)
        btns.addWidget(self.apply_btn)
        layout.addLayout(btns)
        layout.addStretch(1)
        self.preview_btn.clicked.connect(self.previewRequested.emit)
        self.apply_btn.clicked.connect(self.applyRequested.emit)
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        self.select_all_btn.clicked.connect(self.select_all_channels)
        self.unselect_all_btn.clicked.connect(self.unselect_all_channels)

    def set_channels(self, channels: List[str]) -> None:
        self.list_widget.clear()
        for ch in channels:
            item = QtWidgets.QListWidgetItem(ch)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.list_widget.addItem(item)

    def selected_channels(self) -> List[str]:
        chans: List[str] = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                chans.append(item.text())
        return chans

    def select_all_channels(self) -> None:
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item:
                item.setCheckState(QtCore.Qt.CheckState.Checked)

    def unselect_all_channels(self) -> None:
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item:
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def parameters(self, preview: bool = False) -> Dict:
        return {
            "preset": self.preset_combo.currentText(),
            "filter": self.filter_combo.currentText(),
            "window": self.window_spin.value(),
            "polyorder": self.poly_spin.value(),
            "cutoff": self.cutoff_spin.value(),
            "high_cut": self.high_cutoff_spin.value(),
            "order": self.order_spin.value(),
            "target_fs": self.target_fs_spin.value(),
            "apply_selection": self.apply_selection_chk.isChecked(),
            "preview": preview,
        }

    def _apply_preset(self) -> None:
        preset = self.preset_combo.currentText()
        if preset == "Gaze smoothing (savgol 11/2)":
            self.filter_combo.setCurrentText("savgol")
            self.window_spin.setValue(11)
            self.poly_spin.setValue(2)
        elif preset == "Head LPF 6 Hz":
            self.filter_combo.setCurrentText("butter_lowpass")
            self.cutoff_spin.setValue(6.0)
            self.order_spin.setValue(2)
        elif preset == "Resample 60 Hz":
            self.filter_combo.setCurrentText("resample")
            self.target_fs_spin.setValue(60.0)
        elif preset == "Normalize z-score":
            self.filter_combo.setCurrentText("normalize_zscore")


class AnnotationTable(QtWidgets.QTableWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setColumnCount(7)
        self.setHorizontalHeaderLabels(["ID", "Start", "End", "Duration", "Label", "Track", "Color"])
        self.horizontalHeader().setStretchLastSection(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)

    def populate(self, annotations: List[AnnotationSegment]) -> None:
        self.setRowCount(len(annotations))
        for row, ann in enumerate(annotations):
            duration = ann.end - ann.start
            entries = [
                str(ann.id),
                f"{ann.start:.3f}",
                f"{ann.end:.3f}",
                f"{duration:.3f}",
                ann.label,
                ann.track,
                ann.color,
            ]
            for col, val in enumerate(entries):
                self.setItem(row, col, QtWidgets.QTableWidgetItem(val))

    def selected_annotation_id(self) -> int:
        row = self.currentRow()
        if row < 0:
            return -1
        item = self.item(row, 0)
        if item is None:
            return -1
        try:
            return int(item.text())
        except Exception:
            return -1

    def select_annotation(self, ann_id: int) -> None:
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item and item.text().isdigit() and int(item.text()) == ann_id:
                self.setCurrentCell(row, 0)
                self.scrollToItem(item, QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible)
                return


class ExportFigureDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Figure")
        layout = QtWidgets.QFormLayout(self)
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["png", "svg", "pdf"])
        self.dpi_spin = QtWidgets.QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(200)
        self.width_spin = QtWidgets.QDoubleSpinBox()
        self.width_spin.setRange(5.0, 40.0)
        self.width_spin.setValue(15.0)
        self.height_spin = QtWidgets.QDoubleSpinBox()
        self.height_spin.setRange(4.0, 30.0)
        self.height_spin.setValue(10.0)
        layout.addRow("Format", self.format_combo)
        layout.addRow("DPI", self.dpi_spin)
        layout.addRow("Width (cm)", self.width_spin)
        layout.addRow("Height (cm)", self.height_spin)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def export_params(self) -> Dict:
        return {
            "format": self.format_combo.currentText(),
            "dpi": self.dpi_spin.value(),
            "width_cm": self.width_spin.value(),
            "height_cm": self.height_spin.value(),
        }


class PreferencesDialog(QtWidgets.QDialog):
    def __init__(self, current_fs: float, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        layout = QtWidgets.QFormLayout(self)
        self.fs_spin = QtWidgets.QDoubleSpinBox()
        self.fs_spin.setRange(10.0, 1000.0)
        self.fs_spin.setValue(current_fs)
        self.output_dir = QtWidgets.QLineEdit()
        self.output_btn = QtWidgets.QPushButton("Browse…")
        self.output_btn.clicked.connect(self._choose_dir)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.output_dir)
        h.addWidget(self.output_btn)
        layout.addRow("Default sampling rate", self.fs_spin)
        layout.addRow("Default output", h)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def _choose_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output directory")
        if path:
            self.output_dir.setText(path)

    def values(self) -> Dict:
        return {"fs": self.fs_spin.value(), "output_dir": self.output_dir.text()}


class ShortcutDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        layout = QtWidgets.QVBoxLayout(self)
        shortcuts = [
            ("Ctrl+O", "Open CSV"),
            ("Ctrl+S", "Save cleaned CSV"),
            ("Ctrl+Q", "Quit"),
            ("D", "Delete selection"),
            ("M", "Mark bad"),
            ("A", "Annotate selection"),
            ("U", "Undo"),
            ("R", "Redo"),
            ("Space", "Play/Pause"),
            ("Arrow keys", "Scrub time"),
        ]
        for key, desc in shortcuts:
            layout.addWidget(QtWidgets.QLabel(f"<b>{key}</b>: {desc}"))
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


class FrameManagerDialog(QtWidgets.QDialog):
    def __init__(self, frames: Dict[str, Dict], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Coordinate Frames")
        self.frames = frames
        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Frame", "Parent", "Offset (deg)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self._populate()
        btn_add = QtWidgets.QPushButton("Add frame")
        btn_add.clicked.connect(self._add_frame)
        layout.addWidget(btn_add)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def _populate(self) -> None:
        self.table.setRowCount(0)
        for name, info in self.frames.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(info.get("parent", "")))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(info.get("offset", 0.0))))

    def _add_frame(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Frame name", "Name")
        if not ok or not name:
            return
        parent, _ = QtWidgets.QInputDialog.getText(self, "Parent frame", "Parent (optional)")
        offset, _ = QtWidgets.QInputDialog.getDouble(self, "Heading offset", "Offset degrees", decimals=2)
        self.frames[name] = {"parent": parent, "offset": offset}
        self._populate()

    def frames_data(self) -> Dict[str, Dict]:
        return self.frames


class MappingDialog(QtWidgets.QDialog):
    def __init__(self, columns: List[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Mapping")
        layout = QtWidgets.QFormLayout(self)
        self.inputs: Dict[str, QtWidgets.QLineEdit] = {}
        for part in ["Head", "Torso", "Chair", "Left Foot", "Right Foot"]:
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText("x,y,z columns e.g. head_x,head_y,head_z")
            layout.addRow(part, edit)
            self.inputs[part.lower().replace(" ", "_")] = edit
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def mapping(self) -> Dict[str, Dict[str, str]]:
        mapping: Dict[str, Dict[str, str]] = {}
        for key, edit in self.inputs.items():
            txt = edit.text().strip()
            if not txt:
                continue
            parts = [p.strip() for p in txt.split(",") if p.strip()]
            if len(parts) == 3:
                mapping[key] = {"x": parts[0], "y": parts[1], "z": parts[2]}
        return mapping


class CompareTrialsDialog(QtWidgets.QDialog):
    """Overlay a selected channel across multiple trials."""

    def __init__(self, trials: List[str], channels: List[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare Trials")
        self.resize(800, 600)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Select channel to overlay:"))
        self.chan_combo = QtWidgets.QComboBox()
        self.chan_combo.addItems(channels)
        layout.addWidget(self.chan_combo)
        layout.addWidget(QtWidgets.QLabel("Select trials:"))
        self.trial_list = QtWidgets.QListWidget()
        self.trial_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        for t in trials:
            item = QtWidgets.QListWidgetItem(t)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.trial_list.addItem(item)
        layout.addWidget(self.trial_list)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget, 1)
        btns = QtWidgets.QHBoxLayout()
        self.plot_btn = QtWidgets.QPushButton("Plot")
        self.close_btn = QtWidgets.QPushButton("Close")
        btns.addWidget(self.plot_btn)
        btns.addWidget(self.close_btn)
        layout.addLayout(btns)
        self.plot_btn.clicked.connect(self.plot_overlay)
        self.close_btn.clicked.connect(self.accept)

    def selected_trials(self) -> List[str]:
        paths: List[str] = []
        for i in range(self.trial_list.count()):
            item = self.trial_list.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                paths.append(item.text())
        return paths

    def plot_overlay(self) -> None:
        paths = self.selected_trials()
        channel = self.chan_combo.currentText()
        self.plot_widget.clear()
        if not paths or not channel:
            return
        colors = [pg.intColor(i, hues=max(len(paths), 8)) for i in range(len(paths))]
        for idx, p in enumerate(paths):
            try:
                df = pd.read_csv(p)
                if "normalized_time" not in df or channel not in df:
                    continue
                self.plot_widget.plot(df["normalized_time"].values, df[channel].values, pen=colors[idx], name=p)
            except Exception:
                continue


class FilterPreviewDialog(QtWidgets.QDialog):
    def __init__(self, time: np.ndarray, original: np.ndarray, filtered: np.ndarray, channel: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Preview: {channel}")
        self.resize(700, 450)
        n = min(len(time), len(original), len(filtered))
        time = time[:n]
        original = original[:n]
        filtered = filtered[:n]
        layout = QtWidgets.QVBoxLayout(self)
        plot = pg.PlotWidget()
        legend = plot.addLegend()
        try:
            legend.setLabelTextColor((255, 255, 255))
        except Exception:
            pass
        plot.plot(time, original, pen=pg.mkPen('r'), name="Original")
        plot.plot(time, filtered, pen=pg.mkPen('g'), name="Filtered")
        layout.addWidget(plot)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class CalibrationWizard(QtWidgets.QDialog):
    """Estimate frame heading offset using a calibration window."""

    def __init__(self, channels: List[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration Wizard")
        layout = QtWidgets.QFormLayout(self)
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.addItems(channels)
        self.ref_combo = QtWidgets.QComboBox()
        self.ref_combo.addItems(channels)
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setRange(0, 1e6)
        self.start_spin.setDecimals(3)
        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin.setRange(0, 1e6)
        self.end_spin.setDecimals(3)
        layout.addRow("Source heading", self.src_combo)
        layout.addRow("Reference heading", self.ref_combo)
        layout.addRow("Start time (s)", self.start_spin)
        layout.addRow("End time (s)", self.end_spin)
        self.name_edit = QtWidgets.QLineEdit()
        layout.addRow("Save as frame name", self.name_edit)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def params(self) -> Dict:
        return {
            "src": self.src_combo.currentText(),
            "ref": self.ref_combo.currentText(),
            "start": self.start_spin.value(),
            "end": self.end_spin.value(),
            "name": self.name_edit.text().strip(),
        }

