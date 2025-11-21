"""UI dialogs used across the application."""
from __future__ import annotations

from typing import Dict, List, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from data_model import AnnotationSegment
from filter_engine import available_filters


class FilterDialog(QtWidgets.QDialog):
    def __init__(self, channels: List[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Apply Filter")
        self.resize(420, 320)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Select channels:"))
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        for ch in channels:
            item = QtWidgets.QListWidgetItem(ch)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)
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
        form.addRow("Cutoff (Hz)", self.cutoff_spin)
        layout.addLayout(form)
        self.apply_selection_chk = QtWidgets.QCheckBox("Only apply to current selection")
        layout.addWidget(self.apply_selection_chk)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
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
            "filter": self.filter_combo.currentText(),
            "window": self.window_spin.value(),
            "polyorder": self.poly_spin.value(),
            "cutoff": self.cutoff_spin.value(),
            "apply_selection": self.apply_selection_chk.isChecked(),
        }


class AnnotationTable(QtWidgets.QTableWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(["ID", "Start", "End", "Duration", "Label", "Track"])
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
            ]
            for col, val in enumerate(entries):
                self.setItem(row, col, QtWidgets.QTableWidgetItem(val))

    def selected_annotation_id(self) -> int:
        items = self.selectedItems()
        if not items:
            return -1
        try:
            return int(items[0].text())
        except Exception:
            return -1


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

