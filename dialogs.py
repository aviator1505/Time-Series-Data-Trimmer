"""UI dialogs used across the application."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
# Import the Qt binding before pyqtgraph so pyqtgraph binds to PySide6
# even if another Qt binding is installed in the environment.
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QKeySequence, QShortcut
import pyqtgraph as pg

from data_model import AnnotationSegment
from filter_engine import available_filters
from project_manager import load_ui_state, save_ui_state


FILTER_PARAM_MAP: Dict[str, List[str]] = {
    "moving_average": ["window"],
    "median": ["window"],
    "savgol": ["window", "polyorder"],
    "butter_lowpass": ["cutoff", "order"],
    "butter_bandpass": ["low_cut", "high_cut", "order"],
    "detrend": [],
    "resample": ["target_fs"],
    "interpolate": ["method"],
    "derivative": [],
    "integrate": [],
    "normalize_zscore": [],
    "normalize_percent": [],
    "moving_rms": ["window"],
    "absolute": [],
    "invert_polarity": [],
    "invert_mean": [],
    "invert_reference": ["reference"],
    "mirror": ["mirror_mode"],
    "circular_flip": ["wrap_mode"],
    "constant_offset": ["offset"],
}

FILTER_DESCRIPTIONS: Dict[str, str] = {
    "moving_average": "Centered rolling mean smoothing; window is the number of samples.",
    "median": "Rolling median to suppress spikes; window is the number of samples.",
    "savgol": "Savitzky-Golay smoothing that preserves peaks; use an odd window and set polynomial order.",
    "butter_lowpass": "Butterworth low-pass filter; set cutoff frequency in Hz and filter order.",
    "butter_bandpass": "Butterworth band-pass filter; set low/high cutoff frequencies in Hz and filter order.",
    "detrend": "Remove a linear trend from the signal.",
    "resample": "Interpolate the entire trial to a new sampling rate.",
    "interpolate": "Fill gaps using the selected interpolation method.",
    "derivative": "First derivative (rate of change) based on the sample rate.",
    "integrate": "Cumulative integral of the signal.",
    "normalize_zscore": "Normalize to zero mean and unit variance.",
    "normalize_percent": "Scale to +/-100 based on the maximum absolute value.",
    "moving_rms": "Rolling RMS envelope; window controls smoothing in samples.",
    "absolute": "Absolute value of the signal.",
    "invert_polarity": "Negate all values in the signal (-x).",
    "invert_mean": "Flip values around the channel mean (2*mean - x).",
    "invert_reference": "Flip values around a specified reference point (2*ref - x).",
    "mirror": "Flip values around a computed reference: midpoint, median, max, or min of the signal.",
    "circular_flip": "Flip heading/orientation by 180° with circular wrapping (0°→180°, 180°→0°).",
    "constant_offset": "Add/subtract a constant value (e.g., +180 to shift angles).",
}

INTERPOLATE_METHODS = ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]

MIRROR_MODES = ["midpoint", "median", "max", "min", "first"]

WRAP_MODES = ["signed", "unsigned"]


class NumericTableItem(QtWidgets.QTableWidgetItem):
    """Custom QTableWidgetItem that sorts numerically instead of alphabetically.

    This ensures columns containing formatted numbers (e.g., "10.500") sort
    correctly so that 2.000 < 10.500, rather than "10.500" < "2.000" alphabetically.
    """

    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


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
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        form.addRow("Filter type", self.filter_combo)
        self.param_rows: Dict[str, Tuple[QtWidgets.QLabel, QtWidgets.QWidget]] = {}
        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(3, 1001)
        self.window_spin.setValue(11)
        self.window_spin.setToolTip("Window must be odd and > polynomial order")
        self._add_param_row(form, "Window (samples)", self.window_spin, "window")
        self.poly_spin = QtWidgets.QSpinBox()
        self.poly_spin.setRange(1, 5)
        self.poly_spin.setValue(2)
        self.poly_spin.setToolTip("Must be less than window size")
        self._add_param_row(form, "Poly order", self.poly_spin, "polyorder")
        self.cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.cutoff_spin.setRange(0.1, 60.0)
        self.cutoff_spin.setDecimals(2)
        self.cutoff_spin.setValue(6.0)
        self._add_param_row(form, "Low cutoff (Hz)", self.cutoff_spin, "cutoff")
        self.high_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.high_cutoff_spin.setRange(0.1, 60.0)
        self.high_cutoff_spin.setDecimals(2)
        self.high_cutoff_spin.setValue(10.0)
        self._add_param_row(form, "High cutoff (Hz)", self.high_cutoff_spin, "high_cut")
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 6)
        self.order_spin.setValue(2)
        self._add_param_row(form, "Order", self.order_spin, "order")
        self.target_fs_spin = QtWidgets.QDoubleSpinBox()
        self.target_fs_spin.setRange(1.0, 1000.0)
        self.target_fs_spin.setDecimals(2)
        self.target_fs_spin.setValue(60.0)
        self._add_param_row(form, "Target fs (Hz)", self.target_fs_spin, "target_fs")
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(INTERPOLATE_METHODS)
        self._add_param_row(form, "Interpolation", self.method_combo, "method")
        self.reference_spin = QtWidgets.QDoubleSpinBox()
        self.reference_spin.setRange(-1e9, 1e9)
        self.reference_spin.setDecimals(4)
        self.reference_spin.setValue(0.0)
        self._add_param_row(form, "Reference value", self.reference_spin, "reference")
        self.mirror_mode_combo = QtWidgets.QComboBox()
        self.mirror_mode_combo.addItems(MIRROR_MODES)
        self._add_param_row(form, "Mirror around", self.mirror_mode_combo, "mirror_mode")
        self.wrap_mode_combo = QtWidgets.QComboBox()
        self.wrap_mode_combo.addItems(WRAP_MODES)
        self._add_param_row(form, "Output range", self.wrap_mode_combo, "wrap_mode")
        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(-1e9, 1e9)
        self.offset_spin.setDecimals(4)
        self.offset_spin.setValue(0.0)
        self._add_param_row(form, "Offset value", self.offset_spin, "offset")
        layout.addLayout(form)
        self.filter_help = QtWidgets.QLabel()
        self.filter_help.setWordWrap(True)
        layout.addWidget(self.filter_help)
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
        self._on_filter_changed(self.filter_combo.currentText())

    def selected_channels(self) -> List[str]:
        chans: List[str] = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                chans.append(item.text())
        return chans

    def validate_parameters(self) -> List[str]:
        """Validate current filter parameters without modifying UI state.

        Returns:
            List of validation error messages. Empty list if all parameters are valid.
        """
        errors: List[str] = []
        filter_type = self.filter_combo.currentText()

        if filter_type == "savgol":
            window = self.window_spin.value()
            polyorder = self.poly_spin.value()
            if polyorder >= window:
                errors.append(
                    f"Polynomial order ({polyorder}) must be less than window size ({window}). "
                    f"Either increase window or decrease polynomial order."
                )
        elif filter_type == "butter_bandpass":
            low = self.cutoff_spin.value()
            high = self.high_cutoff_spin.value()
            if low >= high:
                errors.append(
                    f"Low cutoff ({low} Hz) must be less than high cutoff ({high} Hz)."
                )

        return errors

    def parameters(self) -> Dict:
        """Get current filter parameters as a dictionary.

        This is a pure getter that does not modify UI state. Window values for
        Savgol filter are adjusted locally (odd requirement) without changing
        the spinbox. Call validate_parameters() first to check for errors.

        Returns:
            Dictionary of filter parameters

        Raises:
            ValueError: If parameters are invalid (polyorder >= window, low >= high cutoff)
        """
        filter_type = self.filter_combo.currentText()

        # Validate parameters - raise errors for invalid combinations
        if filter_type == "savgol":
            window = self.window_spin.value()
            polyorder = self.poly_spin.value()
            if polyorder >= window:
                raise ValueError(
                    f"Polynomial order ({polyorder}) must be less than window size ({window}).\n"
                    f"Either increase window or decrease polynomial order."
                )
        elif filter_type == "butter_bandpass":
            low = self.cutoff_spin.value()
            high = self.high_cutoff_spin.value()
            if low >= high:
                raise ValueError(
                    f"Low cutoff ({low} Hz) must be less than high cutoff ({high} Hz)."
                )

        params: Dict[str, object] = {
            "preset": self.preset_combo.currentText(),
            "filter": filter_type,
            "apply_selection": self.apply_selection_chk.isChecked(),
            "preview": self.preview_requested,
        }
        for key in FILTER_PARAM_MAP.get(filter_type, []):
            params[key] = self._param_value(key)

        # Apply window adjustment locally for savgol (odd window required)
        # This does NOT modify the UI spinbox - the adjustment is only in the returned params
        if filter_type == "savgol" and "window" in params:
            window_val = params["window"]
            if isinstance(window_val, int) and window_val % 2 == 0:
                params["window"] = window_val + 1

        return params

    def _preview(self) -> None:
        self.preview_requested = True
        self.accept()

    def _add_param_row(self, form: QtWidgets.QFormLayout, label_text: str, widget: QtWidgets.QWidget, key: str) -> None:
        label = QtWidgets.QLabel(label_text)
        form.addRow(label, widget)
        self.param_rows[key] = (label, widget)

    def _on_filter_changed(self, filter_type: str) -> None:
        needed = set(FILTER_PARAM_MAP.get(filter_type, []))
        for key, (label, widget) in self.param_rows.items():
            visible = key in needed
            label.setVisible(visible)
            widget.setVisible(visible)
        self.filter_help.setText(FILTER_DESCRIPTIONS.get(filter_type, ""))

    def _param_value(self, key: str):
        if key == "window":
            return self.window_spin.value()
        if key == "polyorder":
            return self.poly_spin.value()
        if key == "cutoff":
            return self.cutoff_spin.value()
        if key == "high_cut":
            return self.high_cutoff_spin.value()
        if key == "low_cut":
            return self.cutoff_spin.value()
        if key == "order":
            return self.order_spin.value()
        if key == "target_fs":
            return self.target_fs_spin.value()
        if key == "method":
            return self.method_combo.currentText()
        if key == "reference":
            return self.reference_spin.value()
        if key == "mirror_mode":
            return self.mirror_mode_combo.currentText()
        if key == "wrap_mode":
            return self.wrap_mode_combo.currentText()
        if key == "offset":
            return self.offset_spin.value()
        return None

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
    applyRequested = QtCore.Signal()
    previewRequested = QtCore.Signal()

    def __init__(self, channels: List[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        # Create splitter for resizable sections
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Channel list section
        list_widget_container = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout(list_widget_container)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(QtWidgets.QLabel("Channels"))
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        list_layout.addWidget(self.list_widget)
        btn_row = QtWidgets.QHBoxLayout()
        self.select_all_btn = QtWidgets.QPushButton("Select all")
        self.unselect_all_btn = QtWidgets.QPushButton("Unselect all")
        btn_row.addWidget(self.select_all_btn)
        btn_row.addWidget(self.unselect_all_btn)
        list_layout.addLayout(btn_row)
        self.splitter.addWidget(list_widget_container)

        # Filter controls section
        form_widget_container = QtWidgets.QWidget()
        form_layout_outer = QtWidgets.QVBoxLayout(form_widget_container)
        form_layout_outer.setContentsMargins(0, 0, 0, 0)
        form_layout_outer.addWidget(QtWidgets.QLabel("Filter Settings"))
        form_layout_outer.addWidget(QtWidgets.QLabel("Preset"))
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
        form_layout_outer.addWidget(self.preset_combo)
        form = QtWidgets.QFormLayout()
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(available_filters())
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        form.addRow("Filter type", self.filter_combo)
        self.param_rows: Dict[str, Tuple[QtWidgets.QLabel, QtWidgets.QWidget]] = {}
        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(3, 1001)
        self.window_spin.setValue(11)
        self.window_spin.setToolTip("Window must be odd and > polynomial order")
        self._add_param_row(form, "Window (samples)", self.window_spin, "window")
        self.poly_spin = QtWidgets.QSpinBox()
        self.poly_spin.setRange(1, 5)
        self.poly_spin.setValue(2)
        self.poly_spin.setToolTip("Must be less than window size")
        self._add_param_row(form, "Poly order", self.poly_spin, "polyorder")
        self.cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.cutoff_spin.setRange(0.1, 60.0)
        self.cutoff_spin.setDecimals(2)
        self.cutoff_spin.setValue(6.0)
        self._add_param_row(form, "Low cutoff (Hz)", self.cutoff_spin, "cutoff")
        self.high_cutoff_spin = QtWidgets.QDoubleSpinBox()
        self.high_cutoff_spin.setRange(0.1, 60.0)
        self.high_cutoff_spin.setDecimals(2)
        self.high_cutoff_spin.setValue(10.0)
        self._add_param_row(form, "High cutoff (Hz)", self.high_cutoff_spin, "high_cut")
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 6)
        self.order_spin.setValue(2)
        self._add_param_row(form, "Order", self.order_spin, "order")
        self.target_fs_spin = QtWidgets.QDoubleSpinBox()
        self.target_fs_spin.setRange(1.0, 1000.0)
        self.target_fs_spin.setDecimals(2)
        self.target_fs_spin.setValue(60.0)
        self._add_param_row(form, "Target fs (Hz)", self.target_fs_spin, "target_fs")
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(INTERPOLATE_METHODS)
        self._add_param_row(form, "Interpolation", self.method_combo, "method")
        self.reference_spin = QtWidgets.QDoubleSpinBox()
        self.reference_spin.setRange(-1e9, 1e9)
        self.reference_spin.setDecimals(4)
        self.reference_spin.setValue(0.0)
        self._add_param_row(form, "Reference value", self.reference_spin, "reference")
        self.mirror_mode_combo = QtWidgets.QComboBox()
        self.mirror_mode_combo.addItems(MIRROR_MODES)
        self._add_param_row(form, "Mirror around", self.mirror_mode_combo, "mirror_mode")
        self.wrap_mode_combo = QtWidgets.QComboBox()
        self.wrap_mode_combo.addItems(WRAP_MODES)
        self._add_param_row(form, "Output range", self.wrap_mode_combo, "wrap_mode")
        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(-1e9, 1e9)
        self.offset_spin.setDecimals(4)
        self.offset_spin.setValue(0.0)
        self._add_param_row(form, "Offset value", self.offset_spin, "offset")
        form_layout_outer.addLayout(form)
        self.filter_help = QtWidgets.QLabel()
        self.filter_help.setWordWrap(True)
        form_layout_outer.addWidget(self.filter_help)
        self.apply_selection_chk = QtWidgets.QCheckBox("Only apply to current selection")
        form_layout_outer.addWidget(self.apply_selection_chk)
        btns = QtWidgets.QHBoxLayout()
        self.preview_btn = QtWidgets.QPushButton("Preview")
        self.apply_btn = QtWidgets.QPushButton("Apply")
        btns.addWidget(self.preview_btn)
        btns.addWidget(self.apply_btn)
        form_layout_outer.addLayout(btns)
        form_layout_outer.addStretch(1)
        self.splitter.addWidget(form_widget_container)

        # Restore saved splitter sizes or use defaults
        ui_state = load_ui_state()
        sizes = ui_state.get("filter_panel_splitter", [200, 300])
        self.splitter.setSizes(sizes)
        self.splitter.splitterMoved.connect(self._save_splitter_state)

        layout.addWidget(self.splitter, 1)

        # Initialize channels and connections
        self.set_channels(channels)
        self.preview_btn.clicked.connect(self.previewRequested.emit)
        self.apply_btn.clicked.connect(self.applyRequested.emit)
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        self.select_all_btn.clicked.connect(self.select_all_channels)
        self.unselect_all_btn.clicked.connect(self.unselect_all_channels)
        self._on_filter_changed(self.filter_combo.currentText())

        # Keyboard shortcuts for improved accessibility
        # Ctrl+Return to apply filter
        apply_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        apply_shortcut.activated.connect(self.applyRequested.emit)
        # Ctrl+P to preview filter
        preview_shortcut = QShortcut(QKeySequence("Ctrl+P"), self)
        preview_shortcut.activated.connect(self.previewRequested.emit)

    def _save_splitter_state(self) -> None:
        """Save splitter positions to UI state."""
        ui_state = load_ui_state()
        ui_state["filter_panel_splitter"] = self.splitter.sizes()
        save_ui_state(ui_state)

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

    def validate_parameters(self) -> List[str]:
        """Validate current filter parameters without modifying UI state.

        Returns:
            List of validation error messages. Empty list if all parameters are valid.
        """
        errors: List[str] = []
        filter_type = self.filter_combo.currentText()

        if filter_type == "savgol":
            window = self.window_spin.value()
            polyorder = self.poly_spin.value()
            if polyorder >= window:
                errors.append(
                    f"Polynomial order ({polyorder}) must be less than window size ({window}). "
                    f"Either increase window or decrease polynomial order."
                )
        elif filter_type == "butter_bandpass":
            low = self.cutoff_spin.value()
            high = self.high_cutoff_spin.value()
            if low >= high:
                errors.append(
                    f"Low cutoff ({low} Hz) must be less than high cutoff ({high} Hz)."
                )

        return errors

    def parameters(self, preview: bool = False) -> Dict:
        """Get current filter parameters as a dictionary.

        This is a pure getter that does not modify UI state. Window values for
        Savgol filter are adjusted locally (odd requirement) without changing
        the spinbox. Call validate_parameters() first to check for errors.

        Args:
            preview: Whether this is a preview request

        Returns:
            Dictionary of filter parameters

        Raises:
            ValueError: If parameters are invalid (polyorder >= window, low >= high cutoff)
        """
        filter_type = self.filter_combo.currentText()

        # Validate parameters - raise errors for invalid combinations
        if filter_type == "savgol":
            window = self.window_spin.value()
            polyorder = self.poly_spin.value()
            if polyorder >= window:
                raise ValueError(
                    f"Polynomial order ({polyorder}) must be less than window size ({window}).\n"
                    f"Either increase window or decrease polynomial order."
                )
        elif filter_type == "butter_bandpass":
            low = self.cutoff_spin.value()
            high = self.high_cutoff_spin.value()
            if low >= high:
                raise ValueError(
                    f"Low cutoff ({low} Hz) must be less than high cutoff ({high} Hz)."
                )

        params: Dict[str, object] = {
            "preset": self.preset_combo.currentText(),
            "filter": filter_type,
            "apply_selection": self.apply_selection_chk.isChecked(),
            "preview": preview,
        }
        for key in FILTER_PARAM_MAP.get(filter_type, []):
            params[key] = self._param_value(key)

        # Apply window adjustment locally for savgol (odd window required)
        # This does NOT modify the UI spinbox - the adjustment is only in the returned params
        if filter_type == "savgol" and "window" in params:
            window_val = params["window"]
            if isinstance(window_val, int) and window_val % 2 == 0:
                params["window"] = window_val + 1

        return params

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

    def _add_param_row(self, form: QtWidgets.QFormLayout, label_text: str, widget: QtWidgets.QWidget, key: str) -> None:
        label = QtWidgets.QLabel(label_text)
        form.addRow(label, widget)
        self.param_rows[key] = (label, widget)

    def _on_filter_changed(self, filter_type: str) -> None:
        needed = set(FILTER_PARAM_MAP.get(filter_type, []))
        for key, (label, widget) in self.param_rows.items():
            visible = key in needed
            label.setVisible(visible)
            widget.setVisible(visible)
        self.filter_help.setText(FILTER_DESCRIPTIONS.get(filter_type, ""))

    def _param_value(self, key: str):
        if key == "window":
            return self.window_spin.value()
        if key == "polyorder":
            return self.poly_spin.value()
        if key == "cutoff":
            return self.cutoff_spin.value()
        if key == "high_cut":
            return self.high_cutoff_spin.value()
        if key == "low_cut":
            return self.cutoff_spin.value()
        if key == "order":
            return self.order_spin.value()
        if key == "target_fs":
            return self.target_fs_spin.value()
        if key == "method":
            return self.method_combo.currentText()
        if key == "reference":
            return self.reference_spin.value()
        if key == "mirror_mode":
            return self.mirror_mode_combo.currentText()
        if key == "wrap_mode":
            return self.wrap_mode_combo.currentText()
        if key == "offset":
            return self.offset_spin.value()
        return None


class AnnotationTable(QtWidgets.QTableWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setColumnCount(7)
        self.setHorizontalHeaderLabels(["ID", "Start", "End", "Duration", "Label", "Track", "Color"])
        self.horizontalHeader().setStretchLastSection(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # Enable sorting by clicking column headers
        self.setSortingEnabled(True)

    def populate(self, annotations: List[AnnotationSegment]) -> None:
        # Temporarily disable sorting during population to avoid performance issues
        # and to prevent sorting from interfering with row insertion
        self.setSortingEnabled(False)
        self.setRowCount(len(annotations))
        for row, ann in enumerate(annotations):
            duration = ann.end - ann.start
            # Column 0: ID (use NumericTableItem for numeric sorting)
            self.setItem(row, 0, NumericTableItem(str(ann.id)))
            # Columns 1-3: Start, End, Duration (use NumericTableItem for numeric sorting)
            self.setItem(row, 1, NumericTableItem(f"{ann.start:.3f}"))
            self.setItem(row, 2, NumericTableItem(f"{ann.end:.3f}"))
            self.setItem(row, 3, NumericTableItem(f"{duration:.3f}"))
            # Columns 4-6: Label, Track, Color (standard string items)
            self.setItem(row, 4, QtWidgets.QTableWidgetItem(ann.label))
            self.setItem(row, 5, QtWidgets.QTableWidgetItem(ann.track))
            self.setItem(row, 6, QtWidgets.QTableWidgetItem(ann.color))
        # Re-enable sorting after population
        self.setSortingEnabled(True)

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
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(["System", "Light", "Dark"])
        layout.addRow("Default sampling rate", self.fs_spin)
        layout.addRow("Default output", h)
        layout.addRow("Theme", self.theme_combo)
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
        return {
            "fs": self.fs_spin.value(),
            "output_dir": self.output_dir.text(),
            "theme": self.theme_combo.currentText(),
        }


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
    """Dialog for managing coordinate frame hierarchy.

    This dialog allows users to define coordinate frames with parent-child
    relationships for proper kinematic chain computation. Each frame has:
    - A unique name
    - An optional parent frame (for hierarchical transforms)
    - A local offset (heading offset relative to parent)
    - A computed total offset (accumulated from all ancestors)

    The hierarchy is essential for multi-sensor setups where:
    - IMU on head is relative to IMU on torso
    - Gaze direction is relative to head orientation
    - Chair reference frame is the "world" frame with offset 0
    """

    def __init__(self, frames: Dict[str, Dict], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Coordinate Frames")
        self.resize(600, 400)
        self.frames = frames
        layout = QtWidgets.QVBoxLayout(self)

        # Info label explaining hierarchy
        info_label = QtWidgets.QLabel(
            "Define coordinate frames with parent relationships. "
            "Total offset is computed by walking the parent chain."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Table with 4 columns: Frame, Parent, Local Offset, Total Offset
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Frame", "Parent", "Local Offset (deg)", "Total Offset (deg)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        # Warning label for cycles (hidden by default)
        self.cycle_warning = QtWidgets.QLabel()
        self.cycle_warning.setStyleSheet("color: red; font-weight: bold;")
        self.cycle_warning.setVisible(False)
        layout.addWidget(self.cycle_warning)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("Add Frame")
        btn_add.clicked.connect(self._add_frame)
        btn_edit = QtWidgets.QPushButton("Edit Frame")
        btn_edit.clicked.connect(self._edit_frame)
        btn_remove = QtWidgets.QPushButton("Remove Frame")
        btn_remove.clicked.connect(self._remove_frame)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_edit)
        btn_layout.addWidget(btn_remove)
        layout.addLayout(btn_layout)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self._populate()

    def _compute_total_offset(self, frame_name: str, visited: set | None = None) -> float:
        """Compute total heading offset by walking parent chain.

        Implements proper kinematic chain: child offset is relative to parent.
        Includes cycle detection to prevent infinite loops.

        Args:
            frame_name: The frame to compute offset for.
            visited: Set of already-visited frame names (for recursion).

        Returns:
            Total accumulated offset in degrees.
        """
        if visited is None:
            visited = set()
        if frame_name in visited:
            return 0.0  # Cycle detected, break recursion
        visited.add(frame_name)

        info = self.frames.get(frame_name, {})
        offset = float(info.get("offset", 0.0))
        parent = info.get("parent", "")

        if parent and parent in self.frames:
            # Recursively add parent's total offset
            offset += self._compute_total_offset(parent, visited)

        return offset

    def _detect_cycle(self, frame_name: str) -> bool:
        """Check if a frame is part of a cycle in the parent chain.

        Args:
            frame_name: The frame to check.

        Returns:
            True if a cycle is detected.
        """
        visited = set()
        current = frame_name
        while current:
            if current in visited:
                return True
            visited.add(current)
            info = self.frames.get(current, {})
            current = info.get("parent", "")
        return False

    def _get_frame_chain(self, frame_name: str) -> List[str]:
        """Get the chain of frames from root to the given frame.

        Args:
            frame_name: The frame to trace.

        Returns:
            List of frame names from root to frame_name.
        """
        chain = []
        current = frame_name
        visited = set()
        while current and current not in visited:
            visited.add(current)
            if current in self.frames:
                chain.append(current)
                current = self.frames[current].get("parent", "")
            else:
                break
        return list(reversed(chain))

    def _populate(self) -> None:
        """Populate the table with frame data including computed total offsets."""
        self.table.setRowCount(0)
        cycles_detected = []

        # Sort frames to show hierarchy (root frames first, then children)
        sorted_frames = self._sort_frames_by_hierarchy()

        for name in sorted_frames:
            info = self.frames[name]
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Check for cycles
            has_cycle = self._detect_cycle(name)
            if has_cycle:
                cycles_detected.append(name)

            # Column 0: Frame name with indentation for hierarchy visualization
            chain = self._get_frame_chain(name)
            indent = "  " * (len(chain) - 1) if chain else ""
            name_item = QtWidgets.QTableWidgetItem(f"{indent}{name}")
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, name)  # Store actual name
            if has_cycle:
                name_item.setBackground(QtGui.QColor(255, 200, 200))
            self.table.setItem(row, 0, name_item)

            # Column 1: Parent
            parent_text = info.get("parent", "")
            parent_item = QtWidgets.QTableWidgetItem(parent_text if parent_text else "(root)")
            if has_cycle:
                parent_item.setBackground(QtGui.QColor(255, 200, 200))
            self.table.setItem(row, 1, parent_item)

            # Column 2: Local Offset
            local_offset = float(info.get("offset", 0.0))
            local_item = NumericTableItem(f"{local_offset:.2f}")
            if has_cycle:
                local_item.setBackground(QtGui.QColor(255, 200, 200))
            self.table.setItem(row, 2, local_item)

            # Column 3: Total Offset (computed)
            if has_cycle:
                total_item = QtWidgets.QTableWidgetItem("CYCLE")
                total_item.setBackground(QtGui.QColor(255, 200, 200))
                total_item.setForeground(QtGui.QColor(180, 0, 0))
            else:
                total_offset = self._compute_total_offset(name)
                total_item = NumericTableItem(f"{total_offset:.2f}")
            self.table.setItem(row, 3, total_item)

        # Show/hide cycle warning
        if cycles_detected:
            self.cycle_warning.setText(
                f"Warning: Cycle detected in frames: {', '.join(cycles_detected)}. "
                "Edit parent relationships to break the cycle."
            )
            self.cycle_warning.setVisible(True)
        else:
            self.cycle_warning.setVisible(False)

    def _sort_frames_by_hierarchy(self) -> List[str]:
        """Sort frames so parents appear before children.

        Returns:
            List of frame names sorted by hierarchy depth.
        """
        # Calculate depth for each frame
        depths: Dict[str, int] = {}
        for name in self.frames:
            depth = 0
            current = name
            visited = set()
            while current and current not in visited:
                visited.add(current)
                info = self.frames.get(current, {})
                parent = info.get("parent", "")
                if parent and parent in self.frames:
                    depth += 1
                    current = parent
                else:
                    break
            depths[name] = depth

        # Sort by depth, then alphabetically
        return sorted(self.frames.keys(), key=lambda x: (depths.get(x, 0), x))

    def _add_frame(self) -> None:
        """Add a new frame with parent selection dialog."""
        dialog = _FrameEditDialog(
            existing_frames=list(self.frames.keys()),
            parent=self
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            name = data["name"]
            if name in self.frames:
                QtWidgets.QMessageBox.warning(
                    self, "Duplicate Name",
                    f"A frame named '{name}' already exists."
                )
                return
            self.frames[name] = {"parent": data["parent"], "offset": data["offset"]}
            self._populate()

    def _edit_frame(self) -> None:
        """Edit the selected frame."""
        row = self.table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, "No Selection", "Please select a frame to edit.")
            return

        name_item = self.table.item(row, 0)
        name = name_item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not name or name not in self.frames:
            return

        info = self.frames[name]
        # Exclude this frame from parent options to prevent self-reference
        other_frames = [f for f in self.frames.keys() if f != name]

        dialog = _FrameEditDialog(
            existing_frames=other_frames,
            current_name=name,
            current_parent=info.get("parent", ""),
            current_offset=float(info.get("offset", 0.0)),
            parent=self
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            new_name = data["name"]

            # Handle rename
            if new_name != name:
                if new_name in self.frames:
                    QtWidgets.QMessageBox.warning(
                        self, "Duplicate Name",
                        f"A frame named '{new_name}' already exists."
                    )
                    return
                # Update any frames that reference the old name as parent
                for other_name, other_info in self.frames.items():
                    if other_info.get("parent") == name:
                        other_info["parent"] = new_name
                del self.frames[name]
                name = new_name

            self.frames[name] = {"parent": data["parent"], "offset": data["offset"]}

            # Check for cycles after edit
            if self._detect_cycle(name):
                QtWidgets.QMessageBox.warning(
                    self, "Cycle Detected",
                    f"Setting '{data['parent']}' as parent of '{name}' creates a cycle. "
                    "The frame will be saved but marked as invalid."
                )

            self._populate()

    def _remove_frame(self) -> None:
        """Remove the selected frame."""
        row = self.table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, "No Selection", "Please select a frame to remove.")
            return

        name_item = self.table.item(row, 0)
        name = name_item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not name or name not in self.frames:
            return

        # Check if other frames reference this as parent
        children = [f for f, info in self.frames.items() if info.get("parent") == name]
        if children:
            reply = QtWidgets.QMessageBox.question(
                self, "Confirm Removal",
                f"Frame '{name}' is the parent of: {', '.join(children)}.\n"
                "Removing it will clear their parent references.\n\nContinue?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            # Clear parent references
            for child in children:
                self.frames[child]["parent"] = ""

        del self.frames[name]
        self._populate()

    def frames_data(self) -> Dict[str, Dict]:
        """Return the frames dictionary."""
        return self.frames


class _FrameEditDialog(QtWidgets.QDialog):
    """Sub-dialog for adding/editing a single frame with parent dropdown."""

    def __init__(
        self,
        existing_frames: List[str],
        current_name: str = "",
        current_parent: str = "",
        current_offset: float = 0.0,
        parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Frame" if current_name else "Add Frame")
        self.resize(350, 200)

        layout = QtWidgets.QFormLayout(self)

        # Frame name
        self.name_edit = QtWidgets.QLineEdit(current_name)
        self.name_edit.setPlaceholderText("e.g., head, torso, chair")
        layout.addRow("Frame Name:", self.name_edit)

        # Parent dropdown with validation
        self.parent_combo = QtWidgets.QComboBox()
        self.parent_combo.addItem("(none - root frame)", "")
        for frame in sorted(existing_frames):
            self.parent_combo.addItem(frame, frame)
        # Set current parent
        if current_parent:
            idx = self.parent_combo.findData(current_parent)
            if idx >= 0:
                self.parent_combo.setCurrentIndex(idx)
        layout.addRow("Parent Frame:", self.parent_combo)

        # Local offset
        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(-360.0, 360.0)
        self.offset_spin.setDecimals(2)
        self.offset_spin.setValue(current_offset)
        self.offset_spin.setSuffix(" deg")
        layout.addRow("Local Offset:", self.offset_spin)

        # Help text
        help_label = QtWidgets.QLabel(
            "Local offset is the heading adjustment relative to the parent frame. "
            "For root frames, this is relative to the lab/world frame."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addRow(help_label)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def _validate_and_accept(self) -> None:
        """Validate input before accepting."""
        name = self.name_edit.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Invalid Name", "Frame name cannot be empty.")
            return
        self.accept()

    def get_data(self) -> Dict:
        """Return the frame data."""
        return {
            "name": self.name_edit.text().strip(),
            "parent": self.parent_combo.currentData(),
            "offset": self.offset_spin.value()
        }


class MappingDialog(QtWidgets.QDialog):
    """Dialog for configuring 3D body part column mappings with validated dropdowns.

    Replaces error-prone text entry with searchable combo boxes that only allow
    selection from actual DataFrame columns. Position fields use separate x/y/z
    dropdowns to prevent invalid entries.
    """

    # Style for invalid/warning state
    INVALID_STYLE = "QComboBox { border: 2px solid #cc0000; background-color: #ffeeee; }"
    VALID_STYLE = ""

    def __init__(self, columns: List[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D Mapping")
        self.columns = sorted(columns)
        self._setup_ui()
        self._auto_detect_mappings()

    def _setup_ui(self) -> None:
        """Build the dialog UI with column combo boxes."""
        layout = QtWidgets.QVBoxLayout(self)

        # Instructions
        instructions = QtWidgets.QLabel(
            "Map body parts to DataFrame columns. Type to filter columns. "
            "Leave empty for unmapped parts."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Scrollable content area for the mapping grid
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        content = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(content)
        grid.setSpacing(8)

        # Headers
        headers = ["Part", "X", "Y", "Z", "Quat X", "Quat Y", "Quat Z", "Quat W", "Dir X", "Dir Y", "Dir Z"]
        for col, hdr in enumerate(headers):
            label = QtWidgets.QLabel(f"<b>{hdr}</b>")
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(label, 0, col)

        # Body part rows
        self.inputs: Dict[str, Dict[str, QtWidgets.QComboBox]] = {}
        parts = ["Head", "Torso", "Chair", "Left Foot", "Right Foot", "Workspace", "Screen"]

        for row, part in enumerate(parts, start=1):
            # Part label
            part_label = QtWidgets.QLabel(part)
            part_label.setMinimumWidth(80)
            grid.addWidget(part_label, row, 0)

            key = part.lower().replace(" ", "_")
            self.inputs[key] = {}

            # Position combos (x, y, z)
            for i, axis in enumerate(["x", "y", "z"]):
                combo = self._create_column_combo(f"Position {axis.upper()}")
                grid.addWidget(combo, row, 1 + i)
                self.inputs[key][axis] = combo

            # Quaternion combos (qx, qy, qz, qw)
            for i, qaxis in enumerate(["qx", "qy", "qz", "qw"]):
                combo = self._create_column_combo(f"Quat {qaxis}")
                grid.addWidget(combo, row, 4 + i)
                self.inputs[key][qaxis] = combo

            # Direction combos (dx, dy, dz)
            for i, daxis in enumerate(["dx", "dy", "dz"]):
                combo = self._create_column_combo(f"Dir {daxis}")
                grid.addWidget(combo, row, 8 + i)
                self.inputs[key][daxis] = combo

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        # Auto-detect button
        auto_btn = QtWidgets.QPushButton("Auto-detect from column names")
        auto_btn.clicked.connect(self._auto_detect_mappings)
        layout.addWidget(auto_btn)

        # Validation status
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #666666;")
        layout.addWidget(self.status_label)

        # Dialog buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Set reasonable dialog size
        self.resize(1100, 450)

    def _create_column_combo(self, placeholder: str = "") -> QtWidgets.QComboBox:
        """Create a searchable combo box populated with DataFrame columns.

        Args:
            placeholder: Placeholder text shown when empty

        Returns:
            Configured QComboBox with type-to-filter functionality
        """
        combo = QtWidgets.QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        combo.setMinimumWidth(100)

        # Add empty option first (means "not mapped")
        combo.addItem("")
        combo.addItems(self.columns)

        # Set placeholder text on the line edit
        if combo.lineEdit():
            combo.lineEdit().setPlaceholderText(placeholder)

        # Configure completer for contains-matching
        completer = combo.completer()
        if completer:
            completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
            completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)

        # Connect change signal for validation feedback
        combo.currentTextChanged.connect(self._on_combo_changed)

        return combo

    def _on_combo_changed(self, text: str) -> None:
        """Handle combo box text changes for validation feedback.

        Args:
            text: Current text in the combo box
        """
        combo = self.sender()
        if not isinstance(combo, QtWidgets.QComboBox):
            return

        # Validate: empty or valid column name
        if text == "" or text in self.columns:
            combo.setStyleSheet(self.VALID_STYLE)
        else:
            combo.setStyleSheet(self.INVALID_STYLE)

        self._update_status()

    def _update_status(self) -> None:
        """Update the status label with mapping summary."""
        mapped_parts = []
        invalid_entries = []

        for part_key, fields in self.inputs.items():
            has_position = all(
                fields[axis].currentText() in self.columns
                for axis in ["x", "y", "z"]
                if fields[axis].currentText()
            )
            pos_filled = any(fields[axis].currentText() for axis in ["x", "y", "z"])

            if pos_filled and has_position:
                # Check if all three position columns are filled
                if all(fields[axis].currentText() for axis in ["x", "y", "z"]):
                    mapped_parts.append(part_key.replace("_", " ").title())

            # Check for invalid entries
            for axis, combo in fields.items():
                text = combo.currentText()
                if text and text not in self.columns:
                    invalid_entries.append(f"{part_key}/{axis}")

        if invalid_entries:
            self.status_label.setText(
                f"Invalid columns: {', '.join(invalid_entries[:3])}{'...' if len(invalid_entries) > 3 else ''}"
            )
            self.status_label.setStyleSheet("color: #cc0000;")
        elif mapped_parts:
            self.status_label.setText(f"Mapped: {', '.join(mapped_parts)}")
            self.status_label.setStyleSheet("color: #006600;")
        else:
            self.status_label.setText("No mappings configured")
            self.status_label.setStyleSheet("color: #666666;")

    def _auto_detect_mappings(self) -> None:
        """Auto-detect column mappings based on common naming patterns.

        Looks for patterns like:
        - Head_x, Head_y, Head_z
        - Mocap_Head_qx, Mocap_Head_qy, etc.
        - left_foot_x, LEFT_FOOT_Y, etc.
        """
        # Define pattern mappings: (part_key, axis) -> list of column name patterns
        patterns = {
            # Position patterns
            ("head", "x"): ["head_x", "Head_x", "HEAD_X", "mocap_head_x", "Mocap_Head_x"],
            ("head", "y"): ["head_y", "Head_y", "HEAD_Y", "mocap_head_y", "Mocap_Head_y"],
            ("head", "z"): ["head_z", "Head_z", "HEAD_Z", "mocap_head_z", "Mocap_Head_z"],
            ("torso", "x"): ["torso_x", "Torso_x", "chest_x", "Chest_x", "TORSO_X"],
            ("torso", "y"): ["torso_y", "Torso_y", "chest_y", "Chest_y", "TORSO_Y"],
            ("torso", "z"): ["torso_z", "Torso_z", "chest_z", "Chest_z", "TORSO_Z"],
            ("chair", "x"): ["chair_x", "Chair_x", "CHAIR_X"],
            ("chair", "y"): ["chair_y", "Chair_y", "CHAIR_Y"],
            ("chair", "z"): ["chair_z", "Chair_z", "CHAIR_Z"],
            ("left_foot", "x"): ["left_foot_x", "Left_Foot_x", "LEFT_FOOT_X", "lfoot_x"],
            ("left_foot", "y"): ["left_foot_y", "Left_Foot_y", "LEFT_FOOT_Y", "lfoot_y"],
            ("left_foot", "z"): ["left_foot_z", "Left_Foot_z", "LEFT_FOOT_Z", "lfoot_z"],
            ("right_foot", "x"): ["right_foot_x", "Right_Foot_x", "RIGHT_FOOT_X", "rfoot_x"],
            ("right_foot", "y"): ["right_foot_y", "Right_Foot_y", "RIGHT_FOOT_Y", "rfoot_y"],
            ("right_foot", "z"): ["right_foot_z", "Right_Foot_z", "RIGHT_FOOT_Z", "rfoot_z"],
        }

        # Also try generic suffix matching for quaternions
        quat_suffixes = ["_qx", "_qy", "_qz", "_qw"]
        dir_suffixes = ["_dx", "_dy", "_dz", "_dir_x", "_dir_y", "_dir_z"]

        # Build a lowercase lookup for case-insensitive matching
        col_lower_map = {c.lower(): c for c in self.columns}

        detected_count = 0

        for (part_key, axis), pattern_list in patterns.items():
            if part_key not in self.inputs or axis not in self.inputs[part_key]:
                continue
            combo = self.inputs[part_key][axis]

            # Try each pattern
            for pattern in pattern_list:
                if pattern in self.columns:
                    combo.setCurrentText(pattern)
                    detected_count += 1
                    break
                # Case-insensitive fallback
                elif pattern.lower() in col_lower_map:
                    combo.setCurrentText(col_lower_map[pattern.lower()])
                    detected_count += 1
                    break

        # Try suffix-based detection for quaternions and directions
        for part_key in self.inputs.keys():
            part_name_variants = [
                part_key,
                part_key.replace("_", ""),
                part_key.title().replace("_", ""),
                part_key.upper().replace("_", "_"),
            ]

            # Quaternions
            for qsuffix in quat_suffixes:
                axis = qsuffix.strip("_")  # "qx", "qy", etc.
                if axis not in self.inputs[part_key]:
                    continue
                combo = self.inputs[part_key][axis]
                if combo.currentText():  # Already set
                    continue

                for variant in part_name_variants:
                    candidate = f"{variant}{qsuffix}"
                    if candidate in self.columns:
                        combo.setCurrentText(candidate)
                        detected_count += 1
                        break
                    elif candidate.lower() in col_lower_map:
                        combo.setCurrentText(col_lower_map[candidate.lower()])
                        detected_count += 1
                        break

        self._update_status()

        if detected_count > 0:
            self.status_label.setText(
                f"Auto-detected {detected_count} column mapping(s). " + self.status_label.text()
            )

    def _validate_and_accept(self) -> None:
        """Validate all entries before accepting the dialog."""
        errors = []

        for part_key, fields in self.inputs.items():
            for axis, combo in fields.items():
                text = combo.currentText()
                if text and text not in self.columns:
                    errors.append(f"{part_key}/{axis}: '{text}' is not a valid column")

            # Check for partial position mappings (some but not all x/y/z)
            pos_filled = [fields[a].currentText() for a in ["x", "y", "z"]]
            filled_count = sum(1 for p in pos_filled if p)
            if 0 < filled_count < 3:
                errors.append(
                    f"{part_key}: Position requires all 3 columns (x, y, z) or none"
                )

            # Check for partial quaternion mappings
            quat_filled = [fields[a].currentText() for a in ["qx", "qy", "qz", "qw"]]
            qfilled_count = sum(1 for q in quat_filled if q)
            if 0 < qfilled_count < 4:
                errors.append(
                    f"{part_key}: Quaternion requires all 4 columns or none"
                )

            # Check for partial direction mappings
            dir_filled = [fields[a].currentText() for a in ["dx", "dy", "dz"]]
            dfilled_count = sum(1 for d in dir_filled if d)
            if 0 < dfilled_count < 3:
                errors.append(
                    f"{part_key}: Direction requires all 3 columns or none"
                )

        if errors:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Mappings",
                "Please fix the following issues:\n\n" + "\n".join(errors[:5]) +
                ("\n..." if len(errors) > 5 else "")
            )
            return

        self.accept()

    def mapping(self) -> Dict[str, Dict[str, str]]:
        """Extract the mapping configuration from the dialog.

        Returns:
            Dictionary mapping part keys to their column assignments.
            Example: {"head": {"x": "Head_x", "y": "Head_y", "z": "Head_z"}}
        """
        result: Dict[str, Dict[str, str]] = {}

        for part_key, fields in self.inputs.items():
            entry: Dict[str, str] = {}

            # Position columns
            for axis in ["x", "y", "z"]:
                col = fields[axis].currentText()
                if col and col in self.columns:
                    entry[axis] = col

            # Quaternion columns
            for axis in ["qx", "qy", "qz", "qw"]:
                col = fields[axis].currentText()
                if col and col in self.columns:
                    entry[axis] = col

            # Direction columns
            for axis in ["dx", "dy", "dz"]:
                col = fields[axis].currentText()
                if col and col in self.columns:
                    entry[axis] = col

            if entry:
                result[part_key] = entry

        return result

    def set_mapping(self, mapping: Dict[str, Dict[str, str]]) -> None:
        """Pre-populate the dialog with an existing mapping.

        Args:
            mapping: Previously saved mapping configuration
        """
        for part_key, columns in mapping.items():
            if part_key not in self.inputs:
                continue

            for axis, col_name in columns.items():
                if axis in self.inputs[part_key]:
                    combo = self.inputs[part_key][axis]
                    if col_name in self.columns:
                        combo.setCurrentText(col_name)

        self._update_status()


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
    """Estimate frame heading offset using a calibration window with real-time preview.

    Features:
    - Preview plot showing source and reference channels
    - Shaded region indicating calibration window
    - Quality metrics (offset, std dev, range, sample count)
    - Quality indicator (green/yellow/red based on std deviation)
    - Integration with current plot selection
    """

    # Quality thresholds for std deviation (in degrees)
    QUALITY_GOOD = 2.0      # Green: < 2 degrees
    QUALITY_MARGINAL = 5.0  # Yellow: 2-5 degrees, Red: >= 5 degrees

    def __init__(
        self,
        channels: List[str],
        df: pd.DataFrame | None = None,
        current_selection: Tuple[float, float] | None = None,
        parent: QtWidgets.QWidget | None = None
    ) -> None:
        """Initialize the calibration wizard.

        Args:
            channels: List of available signal column names
            df: DataFrame containing the data for preview (optional)
            current_selection: Tuple of (start, end) times from plot selection (optional)
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Calibration Wizard")
        self.resize(700, 550)

        self.channels = channels
        self.df = df
        self.current_selection = current_selection

        self._setup_ui()
        self._connect_signals()

        # Initialize preview if we have data
        if self.df is not None and len(self.channels) > 0:
            self._update_preview()

    def _setup_ui(self) -> None:
        """Build the dialog UI with preview plot and metrics."""
        main_layout = QtWidgets.QVBoxLayout(self)

        # Top section: Channel selection and time range
        form_layout = QtWidgets.QFormLayout()

        # Channel selection
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.addItems(self.channels)
        self.ref_combo = QtWidgets.QComboBox()
        self.ref_combo.addItems(self.channels)
        # Default to second channel for reference if available
        if len(self.channels) > 1:
            self.ref_combo.setCurrentIndex(1)

        form_layout.addRow("Source heading:", self.src_combo)
        form_layout.addRow("Reference heading:", self.ref_combo)

        # Time range selection with "Use Current Selection" button
        time_layout = QtWidgets.QHBoxLayout()
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setRange(0, 1e6)
        self.start_spin.setDecimals(3)
        self.start_spin.setSuffix(" s")
        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin.setRange(0, 1e6)
        self.end_spin.setDecimals(3)
        self.end_spin.setSuffix(" s")
        self.end_spin.setValue(1.0)  # Default 1 second window

        time_layout.addWidget(QtWidgets.QLabel("Start:"))
        time_layout.addWidget(self.start_spin)
        time_layout.addWidget(QtWidgets.QLabel("End:"))
        time_layout.addWidget(self.end_spin)

        self.use_selection_btn = QtWidgets.QPushButton("Use Current Selection")
        self.use_selection_btn.setToolTip("Populate start/end from the current plot selection")
        self.use_selection_btn.setEnabled(self.current_selection is not None)
        time_layout.addWidget(self.use_selection_btn)

        form_layout.addRow("Calibration window:", time_layout)

        # Frame name
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("e.g., gaze_vs_head")
        form_layout.addRow("Save as frame name:", self.name_edit)

        main_layout.addLayout(form_layout)

        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        main_layout.addWidget(line)

        # Preview plot
        plot_label = QtWidgets.QLabel("<b>Preview</b>")
        main_layout.addWidget(plot_label)

        self.preview_plot = pg.PlotWidget()
        self.preview_plot.setBackground('w')
        self.preview_plot.showGrid(x=True, y=True, alpha=0.3)
        self.preview_plot.setLabel('bottom', 'Time', units='s')
        self.preview_plot.setLabel('left', 'Heading', units='deg')
        self.preview_plot.addLegend(offset=(10, 10))
        main_layout.addWidget(self.preview_plot, stretch=1)

        # Create plot items
        self.src_curve = self.preview_plot.plot([], [], pen=pg.mkPen(color=(100, 143, 255), width=2), name="Source")
        self.ref_curve = self.preview_plot.plot([], [], pen=pg.mkPen(color=(220, 38, 127), width=2), name="Reference")

        # Calibration window region
        self.cal_region = pg.LinearRegionItem(
            values=(0, 1),
            brush=(100, 200, 100, 50),
            pen=pg.mkPen(color=(50, 150, 50), width=2),
            movable=False
        )
        self.cal_region.setZValue(-10)
        self.preview_plot.addItem(self.cal_region)

        # Quality metrics section
        metrics_group = QtWidgets.QGroupBox("Calibration Quality Metrics")
        metrics_layout = QtWidgets.QGridLayout(metrics_group)

        # Offset
        metrics_layout.addWidget(QtWidgets.QLabel("Offset (mean):"), 0, 0)
        self.offset_label = QtWidgets.QLabel("--")
        self.offset_label.setStyleSheet("font-weight: bold;")
        metrics_layout.addWidget(self.offset_label, 0, 1)

        # Standard deviation
        metrics_layout.addWidget(QtWidgets.QLabel("Std deviation:"), 0, 2)
        self.std_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(self.std_label, 0, 3)

        # Range
        metrics_layout.addWidget(QtWidgets.QLabel("Range (max-min):"), 1, 0)
        self.range_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(self.range_label, 1, 1)

        # Sample count
        metrics_layout.addWidget(QtWidgets.QLabel("Sample count:"), 1, 2)
        self.count_label = QtWidgets.QLabel("--")
        metrics_layout.addWidget(self.count_label, 1, 3)

        # Quality indicator
        metrics_layout.addWidget(QtWidgets.QLabel("Quality:"), 2, 0)
        self.quality_indicator = QtWidgets.QLabel("--")
        metrics_layout.addWidget(self.quality_indicator, 2, 1, 1, 3)

        main_layout.addWidget(metrics_group)

        # Dialog buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        main_layout.addWidget(btns)

    def _connect_signals(self) -> None:
        """Connect UI signals for real-time updates."""
        self.src_combo.currentTextChanged.connect(self._update_preview)
        self.ref_combo.currentTextChanged.connect(self._update_preview)
        self.start_spin.valueChanged.connect(self._update_preview)
        self.end_spin.valueChanged.connect(self._update_preview)
        self.use_selection_btn.clicked.connect(self._use_current_selection)

    def _use_current_selection(self) -> None:
        """Populate time spinboxes from the current plot selection."""
        if self.current_selection is not None:
            start, end = self.current_selection
            # Block signals to avoid multiple preview updates
            self.start_spin.blockSignals(True)
            self.end_spin.blockSignals(True)
            self.start_spin.setValue(start)
            self.end_spin.setValue(end)
            self.start_spin.blockSignals(False)
            self.end_spin.blockSignals(False)
            self._update_preview()

    def _update_preview(self) -> None:
        """Update the preview plot and quality metrics."""
        if self.df is None or self.df.empty:
            self._clear_metrics()
            return

        src_col = self.src_combo.currentText()
        ref_col = self.ref_combo.currentText()

        if not src_col or not ref_col:
            self._clear_metrics()
            return

        if src_col not in self.df.columns or ref_col not in self.df.columns:
            self._clear_metrics()
            return

        # Get time column
        time_col = None
        if 'normalized_time' in self.df.columns:
            time_col = 'normalized_time'
        else:
            for col in self.df.columns:
                if 'time' in col.lower():
                    time_col = col
                    break

        if time_col is None:
            # Use index as time
            time_data = np.arange(len(self.df))
        else:
            time_data = self.df[time_col].values

        src_data = self.df[src_col].values
        ref_data = self.df[ref_col].values

        # Update plot curves
        self.src_curve.setData(time_data, src_data)
        self.ref_curve.setData(time_data, ref_data)

        # Update calibration region
        start_time = self.start_spin.value()
        end_time = self.end_spin.value()
        self.cal_region.setRegion((start_time, end_time))

        # Compute metrics for the calibration window
        mask = (time_data >= start_time) & (time_data <= end_time)
        window_src = src_data[mask]
        window_ref = ref_data[mask]

        if len(window_src) == 0 or len(window_ref) == 0:
            self._clear_metrics()
            return

        # Compute difference (source - reference)
        diff = window_src - window_ref

        # Handle NaN values
        valid_diff = diff[~np.isnan(diff)]
        if len(valid_diff) == 0:
            self._clear_metrics()
            return

        offset = float(np.mean(valid_diff))
        std_dev = float(np.std(valid_diff))
        value_range = float(np.max(valid_diff) - np.min(valid_diff))
        sample_count = len(valid_diff)

        # Update metric labels
        self.offset_label.setText(f"{offset:.3f} deg")
        self.std_label.setText(f"{std_dev:.3f} deg")
        self.range_label.setText(f"{value_range:.3f} deg")
        self.count_label.setText(str(sample_count))

        # Update quality indicator
        self._update_quality_indicator(std_dev)

    def _clear_metrics(self) -> None:
        """Reset all metric displays to default state."""
        self.offset_label.setText("--")
        self.std_label.setText("--")
        self.range_label.setText("--")
        self.count_label.setText("--")
        self.quality_indicator.setText("No data in window")
        self.quality_indicator.setStyleSheet("")

    def _update_quality_indicator(self, std_dev: float) -> None:
        """Update the quality indicator based on standard deviation.

        Args:
            std_dev: Standard deviation of the difference signal in degrees
        """
        if std_dev < self.QUALITY_GOOD:
            # Green - good quality
            color = "#22aa22"
            dot = "<span style='color: #22aa22; font-size: 16px;'>&#9679;</span>"
            text = f"{dot} Good (std < {self.QUALITY_GOOD} deg)"
        elif std_dev < self.QUALITY_MARGINAL:
            # Yellow - marginal quality
            color = "#cc9900"
            dot = "<span style='color: #cc9900; font-size: 16px;'>&#9679;</span>"
            text = f"{dot} Marginal ({self.QUALITY_GOOD}-{self.QUALITY_MARGINAL} deg)"
        else:
            # Red - poor quality
            color = "#cc2222"
            dot = "<span style='color: #cc2222; font-size: 16px;'>&#9679;</span>"
            text = f"{dot} Poor (std >= {self.QUALITY_MARGINAL} deg)"

        self.quality_indicator.setText(text)
        self.quality_indicator.setStyleSheet(f"font-weight: bold;")

    def params(self) -> Dict:
        """Return the calibration parameters.

        Returns:
            Dictionary with keys: src, ref, start, end, name
        """
        return {
            "src": self.src_combo.currentText(),
            "ref": self.ref_combo.currentText(),
            "start": self.start_spin.value(),
            "end": self.end_spin.value(),
            "name": self.name_edit.text().strip(),
        }


class MultiTrialPreviewDialog(QtWidgets.QDialog):
    """
    Preview dialog for multi-file trial selection.
    Shows parsed metadata in an editable table before confirmation.
    """

    def __init__(
        self,
        file_paths: List[str],
        parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Multiple Trials")
        self.resize(950, 500)
        self.file_paths = file_paths
        self._setup_ui()
        self._populate_table()

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)

        # Header
        header = QtWidgets.QLabel(f"Review {len(self.file_paths)} file(s) before adding:")
        header.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(header)

        # Info label
        info = QtWidgets.QLabel(
            "Edit any values as needed. Files that could not be parsed are highlighted."
        )
        layout.addWidget(info)

        # Table
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Filename", "Participant", "Condition",
            "Trial #", "Session", "Angle", "Status"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _populate_table(self) -> None:
        """Parse filenames and fill the table."""
        from project_manager import parse_trial_filename

        self.table.setRowCount(len(self.file_paths))

        for row, path in enumerate(self.file_paths):
            parsed = parse_trial_filename(path)

            # Column 0: Filename (read-only)
            filename_item = QtWidgets.QTableWidgetItem(parsed.original_filename)
            filename_item.setFlags(
                filename_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable
            )
            filename_item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            self.table.setItem(row, 0, filename_item)

            # Column 1: Participant (editable)
            self.table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(parsed.participant)
            )

            # Column 2: Condition (combo box)
            combo = QtWidgets.QComboBox()
            combo.addItems(["", "Sit", "Stand", "Swivel", "Other"])
            if parsed.condition in ["Sit", "Stand", "Swivel"]:
                combo.setCurrentText(parsed.condition)
            self.table.setCellWidget(row, 2, combo)

            # Column 3: Trial # (editable)
            trial_text = str(parsed.trial_number) if parsed.trial_number else ""
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(trial_text))

            # Column 4: Session (editable)
            session_text = str(parsed.session) if parsed.session else ""
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(session_text))

            # Column 5: Angle (editable)
            angle_text = str(parsed.angle) if parsed.angle else ""
            self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(angle_text))

            # Column 6: Status (read-only)
            status = "Parsed" if parsed.parse_success else "Manual entry needed"
            status_item = QtWidgets.QTableWidgetItem(status)
            status_item.setFlags(
                status_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable
            )
            if not parsed.parse_success:
                status_item.setBackground(QtGui.QColor(255, 255, 200))
                # Also highlight the row
                for col in range(6):
                    item = self.table.item(row, col)
                    if item:
                        item.setBackground(QtGui.QColor(255, 255, 200))
            self.table.setItem(row, 6, status_item)

    def get_trial_entries(self) -> List:
        """Collect edited values from the table and return TrialEntry objects."""
        from project_manager import TrialEntry

        entries = []
        for row in range(self.table.rowCount()):
            path = self.table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            participant = self.table.item(row, 1).text().strip()

            # Get condition from combo box
            combo = self.table.cellWidget(row, 2)
            condition = combo.currentText() if combo else ""

            # Parse numeric fields with fallback
            try:
                trial_number = int(self.table.item(row, 3).text())
            except (ValueError, AttributeError):
                trial_number = 0

            try:
                session = int(self.table.item(row, 4).text())
            except (ValueError, AttributeError):
                session = 0

            try:
                angle = int(self.table.item(row, 5).text())
            except (ValueError, AttributeError):
                angle = 0

            entries.append(TrialEntry(
                path=path,
                participant=participant,
                condition=condition,
                trial_number=trial_number,
                session=session,
                angle=angle,
                status="unloaded"
            ))

        return entries


class ExportCSVDialog(QtWidgets.QDialog):
    """Dialog for configuring CSV export with annotation embedding."""

    def __init__(
        self,
        annotations: List[AnnotationSegment],
        parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export CSV with Annotations")
        self.resize(700, 400)
        self.annotations = annotations
        self.manual_indices: Dict[int, int] = {}

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Checkbox to enable annotation embedding
        self.embed_chk = QtWidgets.QCheckBox("Embed annotations as episode columns")
        self.embed_chk.setChecked(True)
        self.embed_chk.stateChanged.connect(self._on_embed_changed)
        layout.addWidget(self.embed_chk)

        # Info label
        info = QtWidgets.QLabel(
            "Annotations will be written as episode_index, episode_type, and episode_state columns."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Episode index mode
        self.mode_group = QtWidgets.QGroupBox("Episode Index Assignment")
        mode_layout = QtWidgets.QVBoxLayout(self.mode_group)

        self.auto_radio = QtWidgets.QRadioButton("Auto-detect (assign sequential indices by start time)")
        self.auto_radio.setChecked(True)
        self.manual_radio = QtWidgets.QRadioButton("Manual configuration")
        self.manual_radio.toggled.connect(self._on_mode_changed)

        mode_layout.addWidget(self.auto_radio)
        mode_layout.addWidget(self.manual_radio)
        layout.addWidget(self.mode_group)

        # Manual index table
        self.table_group = QtWidgets.QGroupBox("Manual Episode Indices")
        table_layout = QtWidgets.QVBoxLayout(self.table_group)

        self.index_table = QtWidgets.QTableWidget(0, 5)
        self.index_table.setHorizontalHeaderLabels([
            "ID", "Start", "End", "Label", "Episode Index"
        ])
        self.index_table.horizontalHeader().setStretchLastSection(True)
        self.index_table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        table_layout.addWidget(self.index_table)

        # Populate table
        self._populate_table()

        self.table_group.setVisible(False)
        layout.addWidget(self.table_group)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _populate_table(self) -> None:
        """Populate table with annotations sorted by start time."""
        sorted_anns = sorted(self.annotations, key=lambda a: a.start)
        self.index_table.setRowCount(len(sorted_anns))

        for row, ann in enumerate(sorted_anns):
            # ID (read-only)
            id_item = QtWidgets.QTableWidgetItem(str(ann.id))
            id_item.setFlags(id_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.index_table.setItem(row, 0, id_item)

            # Start (read-only)
            start_item = QtWidgets.QTableWidgetItem(f"{ann.start:.3f}")
            start_item.setFlags(start_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.index_table.setItem(row, 1, start_item)

            # End (read-only)
            end_item = QtWidgets.QTableWidgetItem(f"{ann.end:.3f}")
            end_item.setFlags(end_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.index_table.setItem(row, 2, end_item)

            # Label (read-only)
            label_item = QtWidgets.QTableWidgetItem(ann.label)
            label_item.setFlags(label_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.index_table.setItem(row, 3, label_item)

            # Episode Index (editable spinbox)
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 9999)
            spin.setValue(row + 1)  # Default: sequential
            spin.setProperty("ann_id", ann.id)
            self.index_table.setCellWidget(row, 4, spin)

    def _on_embed_changed(self, state: int) -> None:
        """Toggle visibility of mode options."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self.mode_group.setEnabled(enabled)
        if not enabled:
            self.table_group.setVisible(False)
            self.resize(700, 400)

    def _on_mode_changed(self, manual: bool) -> None:
        """Show/hide manual index table."""
        self.table_group.setVisible(manual)
        if manual:
            self.resize(700, 600)
        else:
            self.resize(700, 400)

    def _validate_and_accept(self) -> None:
        """Validate indices and check for conflicts before accepting."""
        if self.embed_chk.isChecked() and self.manual_radio.isChecked():
            self.manual_indices = {}
            used_indices: set = set()

            for row in range(self.index_table.rowCount()):
                spin = self.index_table.cellWidget(row, 4)
                ann_id = spin.property("ann_id")
                idx = spin.value()

                if idx in used_indices:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Duplicate Index",
                        f"Episode index {idx} is used multiple times. Each annotation must have a unique index."
                    )
                    return

                self.manual_indices[ann_id] = idx
                used_indices.add(idx)

        self.accept()

    def export_params(self) -> Dict:
        """Return export parameters."""
        return {
            "embed_annotations": self.embed_chk.isChecked(),
            "manual_indices": self.manual_indices if self.manual_radio.isChecked() else None,
        }


class RelativeOrientationDialog(QtWidgets.QDialog):
    """Compute relative orientation between two segments.

    Supports two modes:
    - Simple Heading: Computes relative heading between two heading columns
    - Full Quaternion: Computes relative rotation (yaw, pitch, roll) from quaternions

    Features:
    - Mode toggle between simple and quaternion computation
    - Preview plot showing computed relative angles
    - Output channel naming
    - Integration with FilterEngine for quaternion math
    """

    def __init__(
        self,
        columns: List[str],
        df: pd.DataFrame,
        parent: QtWidgets.QWidget | None = None
    ) -> None:
        """Initialize the relative orientation dialog.

        Args:
            columns: List of available signal column names
            df: DataFrame containing the data for computation and preview
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Relative Orientation")
        self.resize(750, 600)

        self.columns = sorted(columns)
        self.df = df
        self.result_yaw: np.ndarray | None = None
        self.result_pitch: np.ndarray | None = None
        self.result_roll: np.ndarray | None = None
        self.result_heading: np.ndarray | None = None

        self._setup_ui()
        self._connect_signals()
        self._on_mode_changed()

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        main_layout = QtWidgets.QVBoxLayout(self)

        # Mode selection
        mode_group = QtWidgets.QGroupBox("Computation Mode")
        mode_layout = QtWidgets.QHBoxLayout(mode_group)

        self.heading_radio = QtWidgets.QRadioButton("Simple Heading")
        self.heading_radio.setToolTip(
            "Compute relative heading from two heading columns (degrees).\n"
            "Result: (source - target - offset) wrapped to [-180, 180]"
        )
        self.heading_radio.setChecked(True)

        self.quat_radio = QtWidgets.QRadioButton("Full Quaternion")
        self.quat_radio.setToolTip(
            "Compute full 3D relative rotation from quaternion columns.\n"
            "Result: Yaw, Pitch, Roll angles in degrees"
        )

        mode_layout.addWidget(self.heading_radio)
        mode_layout.addWidget(self.quat_radio)
        mode_layout.addStretch()
        main_layout.addWidget(mode_group)

        # Stacked widget for mode-specific inputs
        self.input_stack = QtWidgets.QStackedWidget()

        # --- Simple Heading Page ---
        heading_page = QtWidgets.QWidget()
        heading_layout = QtWidgets.QFormLayout(heading_page)

        self.source_heading_combo = self._create_column_combo("Source heading (degrees)")
        self.target_heading_combo = self._create_column_combo("Target/reference heading")
        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(-360.0, 360.0)
        self.offset_spin.setDecimals(2)
        self.offset_spin.setValue(0.0)
        self.offset_spin.setSuffix(" deg")
        self.heading_output_edit = QtWidgets.QLineEdit()
        self.heading_output_edit.setPlaceholderText("e.g., gaze_vs_head")

        heading_layout.addRow("Source heading:", self.source_heading_combo)
        heading_layout.addRow("Target heading:", self.target_heading_combo)
        heading_layout.addRow("Offset:", self.offset_spin)
        heading_layout.addRow("Output channel name:", self.heading_output_edit)

        heading_help = QtWidgets.QLabel(
            "Computes: ((source - target - offset + 180) % 360) - 180\n"
            "Result is wrapped to [-180, 180] degrees."
        )
        heading_help.setStyleSheet("color: gray; font-size: 10px;")
        heading_help.setWordWrap(True)
        heading_layout.addRow(heading_help)

        self.input_stack.addWidget(heading_page)

        # --- Quaternion Page ---
        quat_page = QtWidgets.QWidget()
        quat_layout = QtWidgets.QVBoxLayout(quat_page)

        # Parent quaternion group
        parent_group = QtWidgets.QGroupBox("Parent Segment Quaternion")
        parent_form = QtWidgets.QFormLayout(parent_group)
        self.parent_qw = self._create_column_combo("qw")
        self.parent_qx = self._create_column_combo("qx")
        self.parent_qy = self._create_column_combo("qy")
        self.parent_qz = self._create_column_combo("qz")
        parent_form.addRow("qw:", self.parent_qw)
        parent_form.addRow("qx:", self.parent_qx)
        parent_form.addRow("qy:", self.parent_qy)
        parent_form.addRow("qz:", self.parent_qz)
        quat_layout.addWidget(parent_group)

        # Child quaternion group
        child_group = QtWidgets.QGroupBox("Child Segment Quaternion")
        child_form = QtWidgets.QFormLayout(child_group)
        self.child_qw = self._create_column_combo("qw")
        self.child_qx = self._create_column_combo("qx")
        self.child_qy = self._create_column_combo("qy")
        self.child_qz = self._create_column_combo("qz")
        child_form.addRow("qw:", self.child_qw)
        child_form.addRow("qx:", self.child_qx)
        child_form.addRow("qy:", self.child_qy)
        child_form.addRow("qz:", self.child_qz)
        quat_layout.addWidget(child_group)

        # Output names
        output_group = QtWidgets.QGroupBox("Output Channel Names")
        output_form = QtWidgets.QFormLayout(output_group)
        self.yaw_output_edit = QtWidgets.QLineEdit()
        self.yaw_output_edit.setPlaceholderText("e.g., relative_yaw")
        self.pitch_output_edit = QtWidgets.QLineEdit()
        self.pitch_output_edit.setPlaceholderText("e.g., relative_pitch")
        self.roll_output_edit = QtWidgets.QLineEdit()
        self.roll_output_edit.setPlaceholderText("e.g., relative_roll")
        output_form.addRow("Yaw (Z rotation):", self.yaw_output_edit)
        output_form.addRow("Pitch (Y rotation):", self.pitch_output_edit)
        output_form.addRow("Roll (X rotation):", self.roll_output_edit)

        quat_help = QtWidgets.QLabel(
            "Computes: child * inverse(parent) using Hamilton product.\n"
            "Leave output name empty to skip that angle."
        )
        quat_help.setStyleSheet("color: gray; font-size: 10px;")
        quat_help.setWordWrap(True)
        output_form.addRow(quat_help)

        quat_layout.addWidget(output_group)
        quat_layout.addStretch()

        self.input_stack.addWidget(quat_page)

        main_layout.addWidget(self.input_stack)

        # Auto-detect button
        auto_btn = QtWidgets.QPushButton("Auto-detect columns")
        auto_btn.clicked.connect(self._auto_detect)
        main_layout.addWidget(auto_btn)

        # Preview section
        preview_group = QtWidgets.QGroupBox("Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)

        self.preview_plot = pg.PlotWidget()
        self.preview_plot.setBackground('w')
        self.preview_plot.showGrid(x=True, y=True, alpha=0.3)
        self.preview_plot.setLabel('bottom', 'Time', units='s')
        self.preview_plot.setLabel('left', 'Angle', units='deg')
        self.preview_plot.addLegend(offset=(10, 10))
        preview_layout.addWidget(self.preview_plot)

        preview_btn_layout = QtWidgets.QHBoxLayout()
        self.compute_btn = QtWidgets.QPushButton("Compute Preview")
        self.compute_btn.clicked.connect(self._compute_preview)
        preview_btn_layout.addWidget(self.compute_btn)
        preview_btn_layout.addStretch()
        preview_layout.addLayout(preview_btn_layout)

        main_layout.addWidget(preview_group, stretch=1)

        # Status label
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        main_layout.addWidget(self.status_label)

        # Dialog buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        main_layout.addWidget(btns)

    def _create_column_combo(self, placeholder: str = "") -> QtWidgets.QComboBox:
        """Create a searchable combo box populated with DataFrame columns."""
        combo = QtWidgets.QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        combo.setMinimumWidth(150)

        combo.addItem("")
        combo.addItems(self.columns)

        if combo.lineEdit():
            combo.lineEdit().setPlaceholderText(placeholder)

        completer = combo.completer()
        if completer:
            completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
            completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)

        return combo

    def _connect_signals(self) -> None:
        """Connect UI signals."""
        self.heading_radio.toggled.connect(self._on_mode_changed)
        self.quat_radio.toggled.connect(self._on_mode_changed)

    def _on_mode_changed(self) -> None:
        """Switch between heading and quaternion input modes."""
        if self.heading_radio.isChecked():
            self.input_stack.setCurrentIndex(0)
        else:
            self.input_stack.setCurrentIndex(1)
        self.preview_plot.clear()
        self.status_label.setText("")

    def _auto_detect(self) -> None:
        """Auto-detect column mappings based on naming patterns."""
        detected = 0
        col_lower_map = {c.lower(): c for c in self.columns}

        if self.heading_radio.isChecked():
            # Detect heading columns
            heading_patterns = ['heading', 'yaw', 'azimuth', 'direction']
            found_headings = []
            for col in self.columns:
                col_l = col.lower()
                if any(p in col_l for p in heading_patterns):
                    found_headings.append(col)

            if len(found_headings) >= 1:
                self.source_heading_combo.setCurrentText(found_headings[0])
                detected += 1
            if len(found_headings) >= 2:
                self.target_heading_combo.setCurrentText(found_headings[1])
                detected += 1

            if not self.heading_output_edit.text() and len(found_headings) >= 2:
                # Suggest output name
                src = found_headings[0].replace('_heading', '').replace('heading', 'src')
                tgt = found_headings[1].replace('_heading', '').replace('heading', 'ref')
                self.heading_output_edit.setText(f"{src}_vs_{tgt}")

        else:
            # Detect quaternion columns
            quat_combos = [
                (self.parent_qw, ['head_qw', 'torso_qw', 'chest_qw', 'parent_qw']),
                (self.parent_qx, ['head_qx', 'torso_qx', 'chest_qx', 'parent_qx']),
                (self.parent_qy, ['head_qy', 'torso_qy', 'chest_qy', 'parent_qy']),
                (self.parent_qz, ['head_qz', 'torso_qz', 'chest_qz', 'parent_qz']),
            ]
            child_combos = [
                (self.child_qw, ['gaze_qw', 'eye_qw', 'child_qw']),
                (self.child_qx, ['gaze_qx', 'eye_qx', 'child_qx']),
                (self.child_qy, ['gaze_qy', 'eye_qy', 'child_qy']),
                (self.child_qz, ['gaze_qz', 'eye_qz', 'child_qz']),
            ]

            for combo, patterns in quat_combos + child_combos:
                for pattern in patterns:
                    if pattern in self.columns:
                        combo.setCurrentText(pattern)
                        detected += 1
                        break
                    elif pattern.lower() in col_lower_map:
                        combo.setCurrentText(col_lower_map[pattern.lower()])
                        detected += 1
                        break

        if detected > 0:
            self.status_label.setText(f"Auto-detected {detected} column(s)")
            self.status_label.setStyleSheet("color: green;")
        else:
            self.status_label.setText("No columns auto-detected")
            self.status_label.setStyleSheet("color: orange;")

    def _compute_preview(self) -> None:
        """Compute and display preview of relative orientation."""
        from filter_engine import FilterEngine

        self.preview_plot.clear()
        self.result_yaw = None
        self.result_pitch = None
        self.result_roll = None
        self.result_heading = None

        if self.df is None or self.df.empty:
            self.status_label.setText("No data available")
            self.status_label.setStyleSheet("color: red;")
            return

        # Get time column
        if 'normalized_time' in self.df.columns:
            time = self.df['normalized_time'].values
        else:
            time = np.arange(len(self.df))

        engine = FilterEngine()

        try:
            if self.heading_radio.isChecked():
                # Simple heading mode
                src = self.source_heading_combo.currentText()
                tgt = self.target_heading_combo.currentText()
                offset = self.offset_spin.value()

                if not src or src not in self.df.columns:
                    self.status_label.setText("Invalid source heading column")
                    self.status_label.setStyleSheet("color: red;")
                    return
                if not tgt or tgt not in self.df.columns:
                    self.status_label.setText("Invalid target heading column")
                    self.status_label.setStyleSheet("color: red;")
                    return

                self.result_heading = engine.relative_heading(self.df, src, tgt, offset)

                # Plot
                self.preview_plot.plot(
                    time, self.result_heading,
                    pen=pg.mkPen(color=(100, 143, 255), width=2),
                    name="Relative Heading"
                )
                self.status_label.setText(
                    f"Mean: {np.nanmean(self.result_heading):.2f} deg, "
                    f"Std: {np.nanstd(self.result_heading):.2f} deg"
                )
                self.status_label.setStyleSheet("color: green;")

            else:
                # Quaternion mode
                pqw = self.parent_qw.currentText()
                pqx = self.parent_qx.currentText()
                pqy = self.parent_qy.currentText()
                pqz = self.parent_qz.currentText()
                cqw = self.child_qw.currentText()
                cqx = self.child_qx.currentText()
                cqy = self.child_qy.currentText()
                cqz = self.child_qz.currentText()

                # Validate all columns exist
                for col_name, label in [
                    (pqw, "Parent qw"), (pqx, "Parent qx"),
                    (pqy, "Parent qy"), (pqz, "Parent qz"),
                    (cqw, "Child qw"), (cqx, "Child qx"),
                    (cqy, "Child qy"), (cqz, "Child qz"),
                ]:
                    if not col_name or col_name not in self.df.columns:
                        self.status_label.setText(f"Invalid column: {label}")
                        self.status_label.setStyleSheet("color: red;")
                        return

                yaw, pitch, roll = engine.relative_rotation(
                    self.df,
                    pqw, pqx, pqy, pqz,
                    cqw, cqx, cqy, cqz
                )
                self.result_yaw = yaw
                self.result_pitch = pitch
                self.result_roll = roll

                # Plot all three
                self.preview_plot.plot(
                    time, yaw,
                    pen=pg.mkPen(color=(100, 143, 255), width=2),
                    name="Yaw"
                )
                self.preview_plot.plot(
                    time, pitch,
                    pen=pg.mkPen(color=(220, 38, 127), width=2),
                    name="Pitch"
                )
                self.preview_plot.plot(
                    time, roll,
                    pen=pg.mkPen(color=(254, 97, 0), width=2),
                    name="Roll"
                )

                self.status_label.setText(
                    f"Yaw: {np.nanmean(yaw):.1f} +/- {np.nanstd(yaw):.1f}, "
                    f"Pitch: {np.nanmean(pitch):.1f} +/- {np.nanstd(pitch):.1f}, "
                    f"Roll: {np.nanmean(roll):.1f} +/- {np.nanstd(roll):.1f}"
                )
                self.status_label.setStyleSheet("color: green;")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    def _validate_and_accept(self) -> None:
        """Validate inputs and accept dialog."""
        if self.heading_radio.isChecked():
            src = self.source_heading_combo.currentText()
            tgt = self.target_heading_combo.currentText()
            output = self.heading_output_edit.text().strip()

            if not src or src not in self.columns:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Input",
                    "Please select a valid source heading column."
                )
                return
            if not tgt or tgt not in self.columns:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Input",
                    "Please select a valid target heading column."
                )
                return
            if not output:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Input",
                    "Please specify an output channel name."
                )
                return

        else:
            # Validate quaternion columns
            quat_cols = [
                (self.parent_qw, "Parent qw"),
                (self.parent_qx, "Parent qx"),
                (self.parent_qy, "Parent qy"),
                (self.parent_qz, "Parent qz"),
                (self.child_qw, "Child qw"),
                (self.child_qx, "Child qx"),
                (self.child_qy, "Child qy"),
                (self.child_qz, "Child qz"),
            ]
            for combo, label in quat_cols:
                col = combo.currentText()
                if not col or col not in self.columns:
                    QtWidgets.QMessageBox.warning(
                        self, "Invalid Input",
                        f"Please select a valid column for {label}."
                    )
                    return

            # At least one output must be specified
            yaw_name = self.yaw_output_edit.text().strip()
            pitch_name = self.pitch_output_edit.text().strip()
            roll_name = self.roll_output_edit.text().strip()
            if not any([yaw_name, pitch_name, roll_name]):
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Input",
                    "Please specify at least one output channel name."
                )
                return

        self.accept()

    def is_heading_mode(self) -> bool:
        """Return True if using simple heading mode."""
        return self.heading_radio.isChecked()

    def params(self) -> Dict:
        """Return the computation parameters.

        For heading mode:
            {"mode": "heading", "source": str, "target": str, "offset": float, "output": str}

        For quaternion mode:
            {"mode": "quaternion",
             "parent": {"qw": str, "qx": str, "qy": str, "qz": str},
             "child": {"qw": str, "qx": str, "qy": str, "qz": str},
             "outputs": {"yaw": str|None, "pitch": str|None, "roll": str|None}}
        """
        if self.heading_radio.isChecked():
            return {
                "mode": "heading",
                "source": self.source_heading_combo.currentText(),
                "target": self.target_heading_combo.currentText(),
                "offset": self.offset_spin.value(),
                "output": self.heading_output_edit.text().strip(),
            }
        else:
            return {
                "mode": "quaternion",
                "parent": {
                    "qw": self.parent_qw.currentText(),
                    "qx": self.parent_qx.currentText(),
                    "qy": self.parent_qy.currentText(),
                    "qz": self.parent_qz.currentText(),
                },
                "child": {
                    "qw": self.child_qw.currentText(),
                    "qx": self.child_qx.currentText(),
                    "qy": self.child_qy.currentText(),
                    "qz": self.child_qz.currentText(),
                },
                "outputs": {
                    "yaw": self.yaw_output_edit.text().strip() or None,
                    "pitch": self.pitch_output_edit.text().strip() or None,
                    "roll": self.roll_output_edit.text().strip() or None,
                },
            }


# ---------------------------------------------------------------------------
# Recipe Preview Dialogs
# ---------------------------------------------------------------------------


class RecipeDataPreviewDialog(QtWidgets.QDialog):
    """Show before/after comparison for a single trial after recipe application.

    Similar to FilterPreviewDialog but with channel selection and statistics.
    """

    def __init__(
        self,
        trial_name: str,
        original_df: pd.DataFrame,
        processed_df: pd.DataFrame,
        signal_columns: List[str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Recipe Preview: {trial_name}")
        self.resize(800, 500)

        self.original_df = original_df
        self.processed_df = processed_df
        self.signal_columns = [c for c in signal_columns if c in original_df.columns and c in processed_df.columns]

        self._setup_ui()
        if self.signal_columns:
            self._update_plot(self.signal_columns[0])

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Channel selector
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Channel:"))
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(self.signal_columns)
        self.channel_combo.currentTextChanged.connect(self._update_plot)
        top_row.addWidget(self.channel_combo, 1)
        layout.addLayout(top_row)

        # Plot widget
        self.plot = pg.PlotWidget()
        legend = self.plot.addLegend()
        try:
            legend.setLabelTextColor((255, 255, 255))
        except Exception:
            pass
        layout.addWidget(self.plot, 1)

        # Statistics panel
        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        # Close button
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _update_plot(self, channel: str) -> None:
        """Update plot and statistics for selected channel."""
        self.plot.clear()

        if channel not in self.original_df.columns or channel not in self.processed_df.columns:
            return

        # Get time column
        time_col = "normalized_time" if "normalized_time" in self.original_df.columns else None
        if time_col is None:
            for col in self.original_df.columns:
                if "time" in col.lower():
                    time_col = col
                    break

        if time_col:
            orig_time = self.original_df[time_col].values
            proc_time = self.processed_df[time_col].values
        else:
            orig_time = np.arange(len(self.original_df))
            proc_time = np.arange(len(self.processed_df))

        orig_data = self.original_df[channel].values
        proc_data = self.processed_df[channel].values

        # Plot
        self.plot.plot(orig_time, orig_data, pen=pg.mkPen('r', width=1), name="Original")
        self.plot.plot(proc_time, proc_data, pen=pg.mkPen('g', width=1), name="Processed")

        # Calculate statistics
        orig_valid = orig_data[~np.isnan(orig_data)]
        proc_valid = proc_data[~np.isnan(proc_data)]

        stats_parts = []
        if len(orig_valid) > 0 and len(proc_valid) > 0:
            orig_mean = np.mean(orig_valid)
            proc_mean = np.mean(proc_valid)
            orig_std = np.std(orig_valid)
            proc_std = np.std(proc_valid)

            stats_parts.append(f"Original: mean={orig_mean:.4f}, std={orig_std:.4f}")
            stats_parts.append(f"Processed: mean={proc_mean:.4f}, std={proc_std:.4f}")
            stats_parts.append(f"Mean change: {proc_mean - orig_mean:+.4f}")

        orig_nan = np.sum(np.isnan(orig_data))
        proc_nan = np.sum(np.isnan(proc_data))
        if orig_nan > 0 or proc_nan > 0:
            stats_parts.append(f"NaN count: {orig_nan} -> {proc_nan}")

        self.stats_label.setText(" | ".join(stats_parts) if stats_parts else "No statistics available")


class RecipePreviewDialog(QtWidgets.QDialog):
    """Preview recipe results before saving with custom output paths.

    Features:
    - Checkboxes for selective trial inclusion
    - Editable output paths with pattern support
    - Preview data button for before/after visualization
    - Validation for duplicate/existing paths
    """

    def __init__(
        self,
        recipe_name: str,
        trial_results: List[Dict],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """
        Args:
            recipe_name: Name of the recipe file (without extension)
            trial_results: List of dicts with keys:
                - path: Original trial path (or "__current__")
                - original_df: DataFrame before recipe
                - processed_df: DataFrame after recipe
                - signal_columns: List of signal column names
                - op_count: Number of operations applied
                - skipped_ops: List of skipped operation descriptions
                - default_output: Default output path
        """
        super().__init__(parent)
        self.setWindowTitle("Recipe Preview")
        self.resize(900, 500)

        self.recipe_name = recipe_name
        self.trial_results = trial_results
        self._manually_edited: set = set()  # Track rows with manual path edits

        self._setup_ui()
        self._populate_table()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Header info
        header = QtWidgets.QLabel(f"<b>Recipe:</b> {self.recipe_name}.json")
        layout.addWidget(header)

        # Pattern row
        pattern_row = QtWidgets.QHBoxLayout()
        pattern_row.addWidget(QtWidgets.QLabel("Output suffix:"))
        self.pattern_edit = QtWidgets.QLineEdit()
        self.pattern_edit.setText(f"_{self.recipe_name}")
        self.pattern_edit.setPlaceholderText("e.g., _cleaned or _{recipe}")
        self.pattern_edit.setToolTip(
            "Suffix added to original filename.\n"
            "Use {recipe} for recipe name, {original} for original filename."
        )
        self.pattern_edit.textChanged.connect(self._on_pattern_changed)
        pattern_row.addWidget(self.pattern_edit, 1)

        self.apply_pattern_btn = QtWidgets.QPushButton("Apply to All")
        self.apply_pattern_btn.clicked.connect(self._apply_pattern_to_all)
        pattern_row.addWidget(self.apply_pattern_btn)
        layout.addLayout(pattern_row)

        # Trial table
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "", "Trial", "Ops", "Skipped", "Output Path"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 30)
        self.table.setColumnWidth(1, 200)
        self.table.setColumnWidth(2, 50)
        self.table.setColumnWidth(3, 70)
        self.table.setSelectionBehavior(QtWidgets.QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)

        # Button row
        btn_row = QtWidgets.QHBoxLayout()

        self.preview_btn = QtWidgets.QPushButton("Preview Data...")
        self.preview_btn.clicked.connect(self._on_preview_data)
        btn_row.addWidget(self.preview_btn)

        self.select_all_btn = QtWidgets.QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        btn_row.addWidget(self.select_all_btn)

        self.select_none_btn = QtWidgets.QPushButton("Select None")
        self.select_none_btn.clicked.connect(self._select_none)
        btn_row.addWidget(self.select_none_btn)

        btn_row.addStretch()

        # Dialog buttons
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self._validate_and_accept)
        btn_row.addWidget(self.apply_btn)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        layout.addLayout(btn_row)

    def _populate_table(self) -> None:
        """Populate table with trial results."""
        import os

        self.table.setRowCount(len(self.trial_results))

        for row, result in enumerate(self.trial_results):
            trial_path = result["path"]
            is_current = trial_path == "__current__"

            # Checkbox
            chk = QtWidgets.QCheckBox()
            chk.setChecked(True)
            chk_widget = QtWidgets.QWidget()
            chk_layout = QtWidgets.QHBoxLayout(chk_widget)
            chk_layout.addWidget(chk)
            chk_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(row, 0, chk_widget)

            # Trial name
            trial_name = "(current session)" if is_current else os.path.basename(trial_path)
            name_item = QtWidgets.QTableWidgetItem(trial_name)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, name_item)

            # Operation count
            op_item = QtWidgets.QTableWidgetItem(str(result["op_count"]))
            op_item.setFlags(op_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            op_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, op_item)

            # Skipped count with tooltip
            skipped = result.get("skipped_ops", [])
            skip_item = QtWidgets.QTableWidgetItem(str(len(skipped)))
            skip_item.setFlags(skip_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            skip_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if skipped:
                skip_item.setToolTip("Skipped:\n" + "\n".join(skipped[:10]))
                skip_item.setBackground(QtGui.QColor(255, 200, 100))  # Light orange
            self.table.setItem(row, 3, skip_item)

            # Output path (editable for file trials, read-only for current)
            if is_current:
                path_item = QtWidgets.QTableWidgetItem("(current session)")
                path_item.setFlags(path_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                path_item.setForeground(QtGui.QColor(128, 128, 128))
            else:
                default_output = result.get("default_output", "")
                path_item = QtWidgets.QTableWidgetItem(default_output)
            self.table.setItem(row, 4, path_item)

        # Track edits to output path
        self.table.itemChanged.connect(self._on_item_changed)

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        """Track manual edits to output paths."""
        if item.column() == 4:
            self._manually_edited.add(item.row())

    def _on_pattern_changed(self, text: str) -> None:
        """Update default output paths when pattern changes (non-edited rows only)."""
        pass  # Pattern is applied via Apply to All button

    def _apply_pattern_to_all(self) -> None:
        """Apply current pattern to all non-current trials."""
        import os

        pattern = self.pattern_edit.text()

        self.table.blockSignals(True)
        for row, result in enumerate(self.trial_results):
            if result["path"] == "__current__":
                continue

            original_path = result["path"]
            base, ext = os.path.splitext(original_path)

            # Apply pattern substitutions
            suffix = pattern
            suffix = suffix.replace("{recipe}", self.recipe_name)
            suffix = suffix.replace("{original}", os.path.basename(base))

            new_path = base + suffix + ext
            self.table.item(row, 4).setText(new_path)

        self._manually_edited.clear()
        self.table.blockSignals(False)

    def _on_preview_data(self) -> None:
        """Show before/after plot for selected trial.

        For memory optimization, original data is loaded on-demand for file trials
        rather than being stored upfront.
        """
        selected_rows = list({idx.row() for idx in self.table.selectedIndexes()})
        if not selected_rows:
            QtWidgets.QMessageBox.information(
                self, "No Selection", "Please select a trial row to preview."
            )
            return

        row = selected_rows[0]
        result = self.trial_results[row]
        trial_path = result["path"]

        # Get original data - either from stored copy or reload from file
        if "original_df" in result:
            # Current session - original was stored
            original_df = result["original_df"]
        else:
            # File trial - reload on-demand to save memory
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                original_df = pd.read_csv(trial_path)
                # Add normalized_time if not present
                if "normalized_time" not in original_df.columns:
                    for col in original_df.columns:
                        if "time" in col.lower():
                            original_df["normalized_time"] = original_df[col]
                            break
                    else:
                        # No time column found, create index-based time
                        original_df["normalized_time"] = np.arange(len(original_df)) / 120.0
            except Exception as e:
                QtWidgets.QApplication.restoreOverrideCursor()
                QtWidgets.QMessageBox.warning(
                    self, "Load Error",
                    f"Failed to reload original data for comparison:\n{e}"
                )
                return
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        dlg = RecipeDataPreviewDialog(
            trial_name=self.table.item(row, 1).text(),
            original_df=original_df,
            processed_df=result["processed_df"],
            signal_columns=result.get("signal_columns", []),
            parent=self,
        )
        dlg.exec()

    def _select_all(self) -> None:
        """Check all trial checkboxes."""
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 0)
            if widget:
                chk = widget.findChild(QtWidgets.QCheckBox)
                if chk:
                    chk.setChecked(True)

    def _select_none(self) -> None:
        """Uncheck all trial checkboxes."""
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 0)
            if widget:
                chk = widget.findChild(QtWidgets.QCheckBox)
                if chk:
                    chk.setChecked(False)

    def _validate_and_accept(self) -> None:
        """Validate paths and accept if valid."""
        import os

        selected = self.get_selected_trials()
        if not selected:
            QtWidgets.QMessageBox.warning(
                self, "No Trials Selected",
                "Please select at least one trial to apply the recipe."
            )
            return

        # Check for duplicate paths
        paths = [r["output_path"] for r in selected if r["path"] != "__current__"]
        duplicates = [p for p in paths if paths.count(p) > 1]
        if duplicates:
            QtWidgets.QMessageBox.warning(
                self, "Duplicate Paths",
                f"Duplicate output paths detected:\n{duplicates[0]}\n\n"
                "Each trial must have a unique output path."
            )
            return

        # Check for existing files
        existing = [p for p in paths if os.path.exists(p)]
        if existing:
            reply = QtWidgets.QMessageBox.question(
                self, "Overwrite Files?",
                f"{len(existing)} output file(s) already exist:\n"
                f"{existing[0]}" + (f"\n... and {len(existing)-1} more" if len(existing) > 1 else "") +
                "\n\nOverwrite?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        self.accept()

    def get_selected_trials(self) -> List[Dict]:
        """Return list of checked trials with their output paths."""
        selected = []
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 0)
            if not widget:
                continue
            chk = widget.findChild(QtWidgets.QCheckBox)
            if not chk or not chk.isChecked():
                continue

            result = self.trial_results[row].copy()
            path_item = self.table.item(row, 4)
            result["output_path"] = path_item.text() if path_item else result.get("default_output", "")
            selected.append(result)

        return selected


class ColumnRenameDialog(QtWidgets.QDialog):
    """Dialog for renaming DataFrame columns.

    Features:
    - Two-column table: Original Name (read-only) | New Name (editable)
    - Validation: no duplicate names, no empty names
    - Reset button to revert all changes
    """

    def __init__(
        self,
        columns: List[str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rename Columns")
        self.resize(500, 400)
        self.original_columns = columns
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Info label
        info = QtWidgets.QLabel(
            "Edit the 'New Name' column to rename channels. "
            "Only changed names will be applied."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Table
        self.table = QtWidgets.QTableWidget(len(self.original_columns), 2)
        self.table.setHorizontalHeaderLabels(["Original Name", "New Name"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )

        for row, col in enumerate(self.original_columns):
            # Original name (read-only)
            orig_item = QtWidgets.QTableWidgetItem(col)
            orig_item.setFlags(orig_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            orig_item.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(row, 0, orig_item)
            # New name (editable, defaults to original)
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(col))

        layout.addWidget(self.table, 1)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()

        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._validate_and_accept)
        btn_row.addWidget(ok_btn)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

    def _reset(self) -> None:
        """Reset all new names to original."""
        for row in range(self.table.rowCount()):
            original = self.table.item(row, 0).text()
            self.table.item(row, 1).setText(original)

    def _validate_and_accept(self) -> None:
        """Validate names and accept if valid."""
        new_names = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 1).text().strip()
            if not name:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Name",
                    f"Row {row + 1}: Column name cannot be empty."
                )
                return
            new_names.append(name)

        # Check for duplicates
        if len(new_names) != len(set(new_names)):
            seen: set = set()
            for name in new_names:
                if name in seen:
                    QtWidgets.QMessageBox.warning(
                        self, "Duplicate Name",
                        f"Column name '{name}' is used multiple times."
                    )
                    return
                seen.add(name)

        self.accept()

    def get_mappings(self) -> Dict[str, str]:
        """Return only columns that were renamed."""
        mappings = {}
        for row in range(self.table.rowCount()):
            old = self.table.item(row, 0).text()
            new = self.table.item(row, 1).text().strip()
            if old != new:
                mappings[old] = new
        return mappings


class ChannelDeleteDialog(QtWidgets.QDialog):
    """Dialog for selecting channels to delete."""

    def __init__(
        self,
        columns: List[str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Delete Channels")
        self.resize(400, 400)
        self.columns = sorted(columns)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Info label
        info = QtWidgets.QLabel("Select channels to delete:")
        layout.addWidget(info)

        # Checkbox list
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        for col in self.columns:
            item = QtWidgets.QListWidgetItem(col)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget, 1)

        # Select All / Select None buttons
        select_row = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        select_row.addWidget(select_all_btn)

        select_none_btn = QtWidgets.QPushButton("Select None")
        select_none_btn.clicked.connect(self._select_none)
        select_row.addWidget(select_none_btn)
        select_row.addStretch()
        layout.addLayout(select_row)

        # Warning label
        warning = QtWidgets.QLabel(
            "<i>Warning: This removes data permanently.<br>"
            "Can be undone with Ctrl+Z.</i>"
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()

        delete_btn = QtWidgets.QPushButton("Delete")
        delete_btn.clicked.connect(self._validate_and_accept)
        btn_row.addWidget(delete_btn)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

    def _select_all(self) -> None:
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(QtCore.Qt.CheckState.Checked)

    def _select_none(self) -> None:
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)

    def _validate_and_accept(self) -> None:
        selected = self.get_selected_columns()
        if not selected:
            QtWidgets.QMessageBox.warning(
                self, "No Selection",
                "Please select at least one channel to delete."
            )
            return
        self.accept()

    def get_selected_columns(self) -> List[str]:
        """Return list of checked columns."""
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                selected.append(item.text())
        return selected


class ChannelDuplicateDialog(QtWidgets.QDialog):
    """Dialog for duplicating channels with new names."""

    def __init__(
        self,
        columns: List[str],
        existing_columns: List[str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Duplicate Channels")
        self.resize(550, 400)
        self.columns = sorted(columns)
        self.existing_columns = set(existing_columns)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Info label
        info = QtWidgets.QLabel(
            "Select channels to duplicate and set new names:"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Table
        self.table = QtWidgets.QTableWidget(len(self.columns), 3)
        self.table.setHorizontalHeaderLabels(["", "Source", "New Name"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Fixed
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.setColumnWidth(0, 30)

        for row, col in enumerate(self.columns):
            # Checkbox
            chk = QtWidgets.QCheckBox()
            chk_widget = QtWidgets.QWidget()
            chk_layout = QtWidgets.QHBoxLayout(chk_widget)
            chk_layout.addWidget(chk)
            chk_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(row, 0, chk_widget)

            # Source name (read-only)
            source_item = QtWidgets.QTableWidgetItem(col)
            source_item.setFlags(source_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            source_item.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(row, 1, source_item)

            # New name (editable)
            new_item = QtWidgets.QTableWidgetItem(f"{col}_copy")
            self.table.setItem(row, 2, new_item)

        layout.addWidget(self.table, 1)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()

        dup_btn = QtWidgets.QPushButton("Duplicate")
        dup_btn.setDefault(True)
        dup_btn.clicked.connect(self._validate_and_accept)
        btn_row.addWidget(dup_btn)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

    def _validate_and_accept(self) -> None:
        mappings = self.get_mappings()
        if not mappings:
            QtWidgets.QMessageBox.warning(
                self, "No Selection",
                "Please select at least one channel to duplicate."
            )
            return

        # Check for empty names
        for source, new_name in mappings.items():
            if not new_name.strip():
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Name",
                    f"New name for '{source}' cannot be empty."
                )
                return

        # Check for duplicate new names
        new_names = list(mappings.values())
        if len(new_names) != len(set(new_names)):
            QtWidgets.QMessageBox.warning(
                self, "Duplicate Names",
                "New names must be unique."
            )
            return

        # Check for conflict with existing columns
        conflicts = [n for n in new_names if n in self.existing_columns]
        if conflicts:
            QtWidgets.QMessageBox.warning(
                self, "Name Conflict",
                f"Column name '{conflicts[0]}' already exists."
            )
            return

        self.accept()

    def get_mappings(self) -> Dict[str, str]:
        """Return {source: new_name} for checked rows."""
        mappings = {}
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 0)
            if not widget:
                continue
            chk = widget.findChild(QtWidgets.QCheckBox)
            if not chk or not chk.isChecked():
                continue

            source = self.table.item(row, 1).text()
            new_name = self.table.item(row, 2).text().strip()
            mappings[source] = new_name

        return mappings


class DerivedChannelDialog(QtWidgets.QDialog):
    """Dialog for creating derived channels from expressions."""

    SAFE_FUNCTIONS = [
        "abs", "sqrt", "sin", "cos", "tan", "log", "log10", "exp", "pow",
        "mean", "std", "min", "max", "sum", "median", "var",
        "floor", "ceil", "round", "clip",
    ]

    def __init__(
        self,
        columns: List[str],
        df: pd.DataFrame,
        existing_columns: List[str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Derived Channel")
        self.resize(700, 550)
        self.columns = sorted(columns)
        self.df = df
        self.existing_columns = set(existing_columns)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Output name
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Output name:"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("e.g., velocity_magnitude")
        name_row.addWidget(self.name_edit, 1)
        layout.addLayout(name_row)

        # Expression
        layout.addWidget(QtWidgets.QLabel("Expression:"))
        self.expr_edit = QtWidgets.QPlainTextEdit()
        self.expr_edit.setPlaceholderText("e.g., sqrt(gaze_x**2 + gaze_y**2)")
        self.expr_edit.setMaximumHeight(80)
        self.expr_edit.textChanged.connect(self._on_expression_changed)
        layout.addWidget(self.expr_edit)

        # Columns and functions side by side
        helpers_layout = QtWidgets.QHBoxLayout()

        # Available columns
        col_group = QtWidgets.QGroupBox("Available Columns (double-click to insert)")
        col_layout = QtWidgets.QVBoxLayout(col_group)
        self.col_list = QtWidgets.QListWidget()
        self.col_list.addItems(self.columns)
        self.col_list.itemDoubleClicked.connect(self._insert_column)
        col_layout.addWidget(self.col_list)
        helpers_layout.addWidget(col_group)

        # Available functions
        func_group = QtWidgets.QGroupBox("Functions (double-click to insert)")
        func_layout = QtWidgets.QVBoxLayout(func_group)
        self.func_list = QtWidgets.QListWidget()
        self.func_list.addItems(self.SAFE_FUNCTIONS)
        self.func_list.itemDoubleClicked.connect(self._insert_function)
        func_layout.addWidget(self.func_list)
        helpers_layout.addWidget(func_group)

        layout.addLayout(helpers_layout, 1)

        # Preview
        preview_group = QtWidgets.QGroupBox("Preview (first 5 rows)")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        self.preview_table = QtWidgets.QTableWidget(5, 2)
        self.preview_table.setHorizontalHeaderLabels(["normalized_time", "result"])
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        self.preview_table.setMaximumHeight(150)
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(preview_group)

        # Status
        self.status_label = QtWidgets.QLabel("Enter an expression to preview")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()

        create_btn = QtWidgets.QPushButton("Create")
        create_btn.setDefault(True)
        create_btn.clicked.connect(self._validate_and_accept)
        btn_row.addWidget(create_btn)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

    def _insert_column(self, item: QtWidgets.QListWidgetItem) -> None:
        """Insert column name at cursor."""
        cursor = self.expr_edit.textCursor()
        cursor.insertText(item.text())
        self.expr_edit.setFocus()

    def _insert_function(self, item: QtWidgets.QListWidgetItem) -> None:
        """Insert function at cursor."""
        cursor = self.expr_edit.textCursor()
        cursor.insertText(f"{item.text()}()")
        # Move cursor inside parentheses
        cursor.movePosition(cursor.MoveOperation.Left)
        self.expr_edit.setTextCursor(cursor)
        self.expr_edit.setFocus()

    def _on_expression_changed(self) -> None:
        """Validate and preview expression."""
        expr = self.expr_edit.toPlainText().strip()
        if not expr:
            self.status_label.setText("Enter an expression to preview")
            self._clear_preview()
            return

        try:
            # Try to evaluate expression
            result = pd.eval(expr, local_dict=self.df.to_dict("series"))

            # Show preview
            self._update_preview(result)
            self.status_label.setText(
                f"<span style='color: green;'>Valid expression. "
                f"Result type: {result.dtype}</span>"
            )
        except Exception as e:
            self.status_label.setText(
                f"<span style='color: red;'>Error: {type(e).__name__}: {str(e)[:80]}</span>"
            )
            self._clear_preview()

    def _update_preview(self, result: pd.Series) -> None:
        """Update preview table with result."""
        n = min(5, len(result))
        self.preview_table.setRowCount(n)

        time_col = self.df.get("normalized_time", pd.Series(range(len(self.df))))

        for i in range(n):
            time_item = QtWidgets.QTableWidgetItem(f"{time_col.iloc[i]:.3f}")
            time_item.setFlags(time_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.preview_table.setItem(i, 0, time_item)

            val = result.iloc[i]
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) and not np.isnan(val) else str(val)
            val_item = QtWidgets.QTableWidgetItem(val_str)
            val_item.setFlags(val_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.preview_table.setItem(i, 1, val_item)

    def _clear_preview(self) -> None:
        """Clear preview table."""
        self.preview_table.setRowCount(0)

    def _validate_and_accept(self) -> None:
        """Validate and accept."""
        name = self.name_edit.text().strip()
        expr = self.expr_edit.toPlainText().strip()

        if not name:
            QtWidgets.QMessageBox.warning(
                self, "Missing Name",
                "Please enter an output name."
            )
            return

        if not expr:
            QtWidgets.QMessageBox.warning(
                self, "Missing Expression",
                "Please enter an expression."
            )
            return

        if name in self.existing_columns:
            QtWidgets.QMessageBox.warning(
                self, "Name Conflict",
                f"Column name '{name}' already exists."
            )
            return

        # Validate expression
        try:
            pd.eval(expr, local_dict=self.df.to_dict("series"))
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Expression",
                f"Expression error:\n{type(e).__name__}: {e}"
            )
            return

        self.accept()

    def get_params(self) -> Dict:
        """Return {"name": str, "expr": str}."""
        return {
            "name": self.name_edit.text().strip(),
            "expr": self.expr_edit.toPlainText().strip(),
        }
