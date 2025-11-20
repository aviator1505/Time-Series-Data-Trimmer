"""
Interactive Time-Series Segmentation, Deletion and Annotation Tool
===============================================================

This application provides an interactive graphical interface built
using PyQt6 and pyqtgraph for exploring and cleaning time‐series
datasets sampled at 120 Hz. Users can load CSV files that follow
the structure of the provided example, select which numeric channels
to view, mark segments on the timeline for deletion or annotation,
and save the cleaned data and annotation metadata. Deleted
segments cause the timeline to collapse and the `normalized_time`
column is recomputed based on the 120 Hz sampling rate. Annotated
segments are preserved in the data but saved separately as JSON.

The application is comprised of four core components:

* DataController – manages loading, cleaning, annotations, and undo/redo.
* SegmentManager – tracks the current selection and stored segments.
* PlotManager – responsible for plotting multiple channels and visual
  feedback for markers, deletions, and annotations.
* MainWindow – constructs the GUI, binds controls to actions and
  coordinates interactions between components.

This file contains the full implementation and can be run as a
standalone script. Dependencies include PyQt6, pyqtgraph, pandas and
NumPy. Make sure these packages are installed in your environment.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


###############################################################################
# Utility classes
###############################################################################

@dataclass
class Annotation:
    """Represents an annotated segment."""
    start: float
    end: float
    label: str


@dataclass
class Deletion:
    """Represents a deleted segment."""
    start: float
    end: float


class DataController(QtCore.QObject):
    """Manages the dataset, deletion and annotation metadata and undo/redo stack."""

    dataChanged = QtCore.pyqtSignal()
    annotationsChanged = QtCore.pyqtSignal()
    statusMessage = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.original_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self.metadata_columns: List[str] = []
        self.signal_columns: List[str] = []
        self.deletions: List[Deletion] = []
        self.annotations: List[Annotation] = []
        # Undo/redo stacks: each element stores (df_copy, deletions_copy, annotations_copy)
        self._undo_stack: List[Tuple[pd.DataFrame, List[Deletion], List[Annotation]]] = []
        self._redo_stack: List[Tuple[pd.DataFrame, List[Deletion], List[Annotation]]] = []
        self.sample_rate: float = 120.0

    def load_csv(self, path: str) -> None:
        """Load a CSV file and classify columns into metadata vs signal."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        # Replace missing values with NaN explicitly
        df = df.replace({"": np.nan, "NaN": np.nan, "nan": np.nan})
        # Determine metadata columns: non-numeric columns (object type)
        metadata_cols = []
        signal_cols = []
        for col in df.columns:
            if col == "normalized_time":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                signal_cols.append(col)
            else:
                metadata_cols.append(col)
        # Save state
        self.original_df = df.copy()
        self.df = df.copy()
        self.metadata_columns = metadata_cols
        self.signal_columns = signal_cols
        self.deletions = []
        self.annotations = []
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.statusMessage.emit(f"Loaded {os.path.basename(path)} with {len(df)} rows and {len(df.columns)} columns")
        self.dataChanged.emit()

    def get_current_df(self) -> pd.DataFrame:
        """Return a copy of the current DataFrame."""
        return self.df.copy() if self.df is not None else pd.DataFrame()

    def push_state(self) -> None:
        """Push the current state onto the undo stack and clear the redo stack."""
        if self.df is None:
            return
        # Deep copy the DataFrame and metadata lists
        self._undo_stack.append((self.df.copy(), list(self.deletions), list(self.annotations)))
        self._redo_stack.clear()

    def undo(self) -> None:
        """Undo the last operation."""
        if not self._undo_stack:
            self.statusMessage.emit("Nothing to undo")
            return
        # Push current state to redo before reverting
        if self.df is not None:
            self._redo_stack.append((self.df.copy(), list(self.deletions), list(self.annotations)))
        df, dels, anns = self._undo_stack.pop()
        self.df = df
        self.deletions = dels
        self.annotations = anns
        self.dataChanged.emit()
        self.annotationsChanged.emit()
        self.statusMessage.emit("Undo last action")

    def redo(self) -> None:
        """Redo the last undone operation."""
        if not self._redo_stack:
            self.statusMessage.emit("Nothing to redo")
            return
        if self.df is not None:
            self._undo_stack.append((self.df.copy(), list(self.deletions), list(self.annotations)))
        df, dels, anns = self._redo_stack.pop()
        self.df = df
        self.deletions = dels
        self.annotations = anns
        self.dataChanged.emit()
        self.annotationsChanged.emit()
        self.statusMessage.emit("Redo last undone action")

    def delete_segment(self, start: float, end: float) -> None:
        """Delete rows within [start, end] and collapse time."""
        if self.df is None:
            return
        if start >= end:
            self.statusMessage.emit("Invalid deletion range")
            return
        # Push state to undo stack
        self.push_state()
        # Perform deletion
        mask = (self.df["normalized_time"] < start) | (self.df["normalized_time"] > end)
        new_df = self.df.loc[mask].reset_index(drop=True)
        # Recompute normalized_time
        n = len(new_df)
        new_df["normalized_time"] = np.arange(n) / self.sample_rate
        self.df = new_df
        # Record deletion
        self.deletions.append(Deletion(start=start, end=end))
        self.dataChanged.emit()
        self.annotationsChanged.emit()
        self.statusMessage.emit(f"Deleted segment {start:.3f}–{end:.3f}s")

    def annotate_segment(self, start: float, end: float, label: str) -> None:
        """Record an annotation without altering the data."""
        if self.df is None:
            return
        if start >= end:
            self.statusMessage.emit("Invalid annotation range")
            return
        # Push state to undo stack so that annotations can be undone
        self.push_state()
        self.annotations.append(Annotation(start=start, end=end, label=label))
        self.annotationsChanged.emit()
        self.statusMessage.emit(f"Annotated segment {start:.3f}–{end:.3f}s as '{label}'")

    def save_cleaned_csv(self, path: str) -> None:
        """Save the current DataFrame to a CSV file."""
        if self.df is None:
            self.statusMessage.emit("No data to save")
            return
        self.df.to_csv(path, index=False)
        self.statusMessage.emit(f"Saved cleaned data to {path}")

    def save_annotations(self, path: str) -> None:
        """Save annotations and deletions to JSON."""
        data = {
            "annotations": [ann.__dict__ for ann in self.annotations],
            "deletions": [del_.__dict__ for del_ in self.deletions],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.statusMessage.emit(f"Saved annotations to {path}")

    def load_annotations(self, path: str) -> None:
        """Load annotations and deletions from a JSON file and apply to current data."""
        if not os.path.isfile(path):
            self.statusMessage.emit(f"Annotation file not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        anns = data.get("annotations", [])
        dels = data.get("deletions", [])
        # Reset current state and reapply
        if self.original_df is not None:
            self.df = self.original_df.copy()
            self.deletions = []
            self.annotations = []
            # Apply deletions first
            for d in sorted(dels, key=lambda x: (x["start"], x["end"])):
                self.delete_segment(d["start"], d["end"])
            # Apply annotations after deletions
            for a in anns:
                self.annotate_segment(a["start"], a["end"], a["label"])
            self.statusMessage.emit(f"Loaded annotations from {path}")
        else:
            self.statusMessage.emit("No data loaded; cannot apply annotations")


class SegmentManager(QtCore.QObject):
    """Handles temporary marker positions for segment selection."""

    selectionChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.start: Optional[float] = None
        self.end: Optional[float] = None

    def set_marker(self, time_point: float) -> None:
        """Assign start or end marker based on what is currently empty."""
        if self.start is None:
            self.start = time_point
        elif self.end is None:
            self.end = time_point
        else:
            # Both markers already set; reset and set new start
            self.start = time_point
            self.end = None
        if self.start is not None and self.end is not None:
            # Emit selection in sorted order
            s, e = sorted([self.start, self.end])
            self.selectionChanged.emit(s, e)

    def clear(self) -> None:
        self.start = None
        self.end = None
        self.selectionChanged.emit(float('nan'), float('nan'))


class PlotManager(QtCore.QObject):
    """Handles plotting of signals and visual feedback on a pyqtgraph canvas."""
    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plots: List[pg.PlotItem] = []
        self.data: pd.DataFrame = pd.DataFrame()
        self.channels: List[str] = []
        self.marker_lines: Tuple[Optional[pg.InfiniteLine], Optional[pg.InfiniteLine]] = (None, None)
        self.annotation_regions: List[pg.LinearRegionItem] = []
        self.deletion_regions: List[pg.LinearRegionItem] = []
        # Colours for annotations and deletions
        self.annotation_brush = pg.mkBrush(QtGui.QColor(200, 200, 255, 100))  # light blue
        self.deletion_brush = pg.mkBrush(QtGui.QColor(255, 200, 200, 150))   # light red

    def set_data(self, df: pd.DataFrame) -> None:
        """Set the dataset for plotting."""
        self.data = df.copy()
        self.update_plots()

    def set_channels(self, channels: List[str]) -> None:
        """Update the list of channels to display."""
        self.channels = channels
        self.update_plots()

    def update_plots(self) -> None:
        """Recreate plots according to current data and channel selection."""
        # Clear existing layout
        for p in self.plots:
            p.clear()
            self.plot_widget.removeItem(p)
        self.plots.clear()
        if self.data.empty or not self.channels:
            return
        time = self.data["normalized_time"].values
        layout = self.plot_widget
        # Add one plot per channel
        for idx, ch in enumerate(self.channels):
            p = layout.addPlot(row=idx, col=0)
            p.clear()
            p.plot(time, self.data[ch].values, pen=pg.mkPen())
            p.showGrid(x=True, y=True)
            p.setLabel("left", ch)
            if idx == len(self.channels) - 1:
                p.setLabel("bottom", "Time (s)")
            # share x-axis
            if idx > 0:
                p.setXLink(self.plots[0])
            self.plots.append(p)
        # Reset marker references and region visuals
        self.marker_lines = (None, None)
        self.clear_regions()

    def clear_regions(self) -> None:
        """Remove all shaded regions from plots."""
        # Remove existing annotation and deletion regions
        for region in self.annotation_regions + self.deletion_regions:
            for plot in self.plots:
                try:
                    plot.removeItem(region)
                except Exception:
                    pass
        self.annotation_regions.clear()
        self.deletion_regions.clear()

    def draw_marker(self, idx: int, time_point: float) -> None:
        """Draw a vertical marker line on all plots. idx=0 for start, idx=1 for end."""
        if not self.plots:
            return
        # Create or update the infinite line
        line = self.marker_lines[idx]
        if line is None:
            line = pg.InfiniteLine(pos=time_point, angle=90, pen=pg.mkPen('b'))
            for p in self.plots:
                p.addItem(line)
        else:
            line.setPos(time_point)
        markers = list(self.marker_lines)
        markers[idx] = line
        self.marker_lines = tuple(markers)

    def clear_markers(self) -> None:
        """Remove marker lines from the plots."""
        for line in self.marker_lines:
            if line is not None:
                for p in self.plots:
                    try:
                        p.removeItem(line)
                    except Exception:
                        pass
        self.marker_lines = (None, None)

    def update_regions(self, annotations: List[Annotation], deletions: List[Deletion]) -> None:
        """Draw shaded regions based on current annotations and deletions."""
        # Clear any existing regions
        self.clear_regions()
        # Draw deletions first
        for del_seg in deletions:
            region = pg.LinearRegionItem(values=(del_seg.start, del_seg.end), brush=self.deletion_brush, movable=False)
            region.setZValue(-10)  # ensure behind data
            self.deletion_regions.append(region)
            for p in self.plots:
                p.addItem(region)
        # Draw annotations
        for ann in annotations:
            region = pg.LinearRegionItem(values=(ann.start, ann.end), brush=self.annotation_brush, movable=False)
            region.setZValue(-5)
            region.annot_label = ann.label  # attach label attribute
            self.annotation_regions.append(region)
            for p in self.plots:
                p.addItem(region)
        # Attach tooltips to annotation regions
        for region in self.annotation_regions:
            def make_tooltip(label: str):
                return lambda: QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), label)
            region.sigClicked.connect(make_tooltip(region.annot_label))  # type: ignore


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with sidebar controls and plotting area."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Time-Series Segment Editor")
        self.resize(1200, 800)
        # Central widget with horizontal layout: sidebar and plot area
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        # Left sidebar
        self.sidebar = QtWidgets.QWidget()
        self.sidebar.setMinimumWidth(300)
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(self.sidebar)
        # Plot area
        self.plot_manager = PlotManager()
        layout.addWidget(self.plot_manager.plot_widget, stretch=1)
        # Instantiate data controller and segment manager
        self.data_ctrl = DataController()
        self.segment_mgr = SegmentManager()
        # Build sidebar UI
        self.build_sidebar(sidebar_layout)
        # Connect signals
        self.connect_signals()

    def build_sidebar(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Assemble the sidebar widgets."""
        # File operations
        self.load_btn = QtWidgets.QPushButton("Load CSV…")
        self.save_btn = QtWidgets.QPushButton("Save Cleaned CSV…")
        self.save_ann_btn = QtWidgets.QPushButton("Save Annotations…")
        self.load_ann_btn = QtWidgets.QPushButton("Load Annotations…")
        layout.addWidget(self.load_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.save_ann_btn)
        layout.addWidget(self.load_ann_btn)
        # Channel selection
        layout.addWidget(QtWidgets.QLabel("Channels:"))
        self.channel_scroll = QtWidgets.QScrollArea()
        self.channel_scroll.setWidgetResizable(True)
        self.channel_container = QtWidgets.QWidget()
        self.channel_layout = QtWidgets.QVBoxLayout(self.channel_container)
        self.channel_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_scroll.setWidget(self.channel_container)
        layout.addWidget(self.channel_scroll, stretch=1)
        # Segment controls
        layout.addWidget(QtWidgets.QLabel("Segment Controls:"))
        # Markers: instructions label
        self.marker_hint = QtWidgets.QLabel("Click on plot to set start/end. Then choose an action.")
        self.marker_hint.setWordWrap(True)
        layout.addWidget(self.marker_hint)
        # Action buttons
        self.delete_btn = QtWidgets.QPushButton("Delete Segment")
        self.annotate_btn = QtWidgets.QPushButton("Annotate Segment")
        # Annotation label dropdown and text entry
        self.label_combo = QtWidgets.QComboBox()
        self.label_combo.setEditable(False)
        self.label_combo.addItems(["blink", "dropout", "spike", "head-movement artefact"])
        self.custom_label_edit = QtWidgets.QLineEdit()
        self.custom_label_edit.setPlaceholderText("Custom label… (leave empty to use selection)")
        layout.addWidget(QtWidgets.QLabel("Annotation label:"))
        layout.addWidget(self.label_combo)
        layout.addWidget(self.custom_label_edit)
        layout.addWidget(self.annotate_btn)
        layout.addWidget(self.delete_btn)
        # Undo/redo and clear
        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.redo_btn = QtWidgets.QPushButton("Redo")
        self.clear_sel_btn = QtWidgets.QPushButton("Clear Selection")
        layout.addWidget(self.undo_btn)
        layout.addWidget(self.redo_btn)
        layout.addWidget(self.clear_sel_btn)
        # Status box
        layout.addWidget(QtWidgets.QLabel("Status:"))
        self.status_box = QtWidgets.QLabel("")
        self.status_box.setWordWrap(True)
        self.status_box.setMinimumHeight(40)
        layout.addWidget(self.status_box)
        # Spacer
        layout.addStretch(1)

    def connect_signals(self) -> None:
        """Connect signals between controllers and UI widgets."""
        # File operations
        self.load_btn.clicked.connect(self.on_load_csv)
        self.save_btn.clicked.connect(self.on_save_csv)
        self.save_ann_btn.clicked.connect(self.on_save_annotations)
        self.load_ann_btn.clicked.connect(self.on_load_annotations)
        # Segment actions
        self.delete_btn.clicked.connect(self.on_delete_segment)
        self.annotate_btn.clicked.connect(self.on_annotate_segment)
        self.undo_btn.clicked.connect(self.data_ctrl.undo)
        self.redo_btn.clicked.connect(self.data_ctrl.redo)
        self.clear_sel_btn.clicked.connect(self.segment_mgr.clear)
        # Data controller signals
        self.data_ctrl.dataChanged.connect(self.on_data_changed)
        self.data_ctrl.annotationsChanged.connect(self.on_annotations_changed)
        self.data_ctrl.statusMessage.connect(self.update_status)
        # Segment manager signals
        self.segment_mgr.selectionChanged.connect(self.on_selection_changed)
        # Plot interactions: capture mouse clicks
        self.plot_manager.plot_widget.scene().sigMouseClicked.connect(self.on_plot_clicked)

    def on_load_csv(self) -> None:
        """Handle loading of a CSV file via dialog."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV file", "", "CSV files (*.csv)")
        if path:
            try:
                self.data_ctrl.load_csv(path)
                # Populate channels list
                self.populate_channels()
            except Exception as e:
                self.update_status(f"Error loading file: {e}")

    def on_save_csv(self) -> None:
        """Save cleaned CSV via dialog."""
        if self.data_ctrl.df is None:
            self.update_status("No data to save")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save cleaned CSV", "", "CSV files (*.csv)")
        if path:
            # Ensure .csv extension
            if not path.lower().endswith(".csv"):
                path += ".csv"
            self.data_ctrl.save_cleaned_csv(path)

    def on_save_annotations(self) -> None:
        """Save annotations and deletions via dialog."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save annotations", "", "JSON files (*.json)")
        if path:
            if not path.lower().endswith(".json"):
                path += ".json"
            self.data_ctrl.save_annotations(path)

    def on_load_annotations(self) -> None:
        """Load annotations from file and apply."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load annotations", "", "JSON files (*.json)")
        if path:
            self.data_ctrl.load_annotations(path)

    def populate_channels(self) -> None:
        """Populate channel checkboxes based on loaded data."""
        # Clear existing checkboxes
        for i in reversed(range(self.channel_layout.count())):
            item = self.channel_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        # Create a checkbox for each signal column
        for col in self.data_ctrl.signal_columns:
            cb = QtWidgets.QCheckBox(col)
            cb.stateChanged.connect(self.on_channel_toggled)
            self.channel_layout.addWidget(cb)
        # Check first few by default
        for idx in range(min(5, self.channel_layout.count())):
            item = self.channel_layout.itemAt(idx)
            cb = item.widget()
            if isinstance(cb, QtWidgets.QCheckBox):
                cb.setChecked(True)

    def on_channel_toggled(self) -> None:
        """Update plots when a channel checkbox state changes."""
        channels = []
        for i in range(self.channel_layout.count()):
            item = self.channel_layout.itemAt(i)
            cb = item.widget()
            if isinstance(cb, QtWidgets.QCheckBox) and cb.isChecked():
                channels.append(cb.text())
        self.plot_manager.set_channels(channels)
        # Immediately update regions and markers when channels change
        self.plot_manager.update_regions(self.data_ctrl.annotations, self.data_ctrl.deletions)
        # Clear any selection
        self.segment_mgr.clear()

    def on_data_changed(self) -> None:
        """Refresh plot and channel list when data changes."""
        df = self.data_ctrl.get_current_df()
        self.plot_manager.set_data(df)
        # Keep current channel selection if possible
        current_channels = self.plot_manager.channels
        # Remove channels that are no longer valid
        valid = [ch for ch in current_channels if ch in df.columns]
        self.plot_manager.set_channels(valid)
        # Reset selection markers
        self.segment_mgr.clear()

    def on_annotations_changed(self) -> None:
        """Update shaded regions when annotations or deletions change."""
        self.plot_manager.update_regions(self.data_ctrl.annotations, self.data_ctrl.deletions)

    def update_status(self, msg: str) -> None:
        """Display status messages to the user."""
        self.status_box.setText(msg)

    def on_plot_clicked(self, event: Any) -> None:
        """Handle click events in the plot scene to set markers."""
        if not event.scenePos():
            return
        # Map scene position to the data's x-coordinate
        pos = event.scenePos()
        # Only consider clicks inside the first plot; ignore label clicks
        if not self.plot_manager.plots:
            return
        view_box = self.plot_manager.plots[0].getViewBox()
        if view_box.mapRectFromScene(pos.toPoint()).contains(pos):
            x_val = view_box.mapSceneToView(pos).x()
            # Snap to nearest sample time for accuracy
            x_val = float(x_val)
            # Update markers through SegmentManager
            self.segment_mgr.set_marker(x_val)

    def on_selection_changed(self, start: float, end: float) -> None:
        """When both markers are set, draw lines and optionally update UI."""
        if np.isnan(start) or np.isnan(end):
            # Selection cleared
            self.plot_manager.clear_markers()
        else:
            # Draw markers
            self.plot_manager.draw_marker(0, start)
            self.plot_manager.draw_marker(1, end)
            # Optionally show in status
            self.update_status(f"Selected segment: {start:.3f}–{end:.3f}s")

    def on_delete_segment(self) -> None:
        """Trigger deletion of currently selected segment."""
        if self.segment_mgr.start is None or self.segment_mgr.end is None:
            self.update_status("Select start and end points first")
            return
        s, e = sorted([self.segment_mgr.start, self.segment_mgr.end])
        self.data_ctrl.delete_segment(s, e)
        # After deletion, clear selection markers
        self.segment_mgr.clear()

    def on_annotate_segment(self) -> None:
        """Trigger annotation of currently selected segment with chosen label."""
        if self.segment_mgr.start is None or self.segment_mgr.end is None:
            self.update_status("Select start and end points first")
            return
        s, e = sorted([self.segment_mgr.start, self.segment_mgr.end])
        # Determine label: if custom field non-empty use that; else dropdown
        custom = self.custom_label_edit.text().strip()
        label = custom if custom else self.label_combo.currentText()
        self.data_ctrl.annotate_segment(s, e, label)
        # Clear selection markers but keep drawn regions
        self.segment_mgr.clear()


def main() -> None:
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()