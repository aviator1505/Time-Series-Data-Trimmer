"""2D plotting controller built on pyqtgraph."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt6 import QtGui
import pyqtgraph as pg

from data_model import AnnotationSegment


class PlotController2D:
    def __init__(self) -> None:
        self.widget = pg.GraphicsLayoutWidget()
        self.plots: List[pg.PlotItem] = []
        self.channels: List[str] = []
        self.data: pd.DataFrame = pd.DataFrame()
        self.selection_region: Optional[pg.LinearRegionItem] = None
        self.time_cursor: Optional[pg.InfiniteLine] = None
        self.annotation_regions: List[pg.LinearRegionItem] = []
        self.annotation_clone_regions: List[pg.LinearRegionItem] = []
        self.deletion_regions: List[pg.LinearRegionItem] = []
        self.marker_color = (80, 80, 220)
        self.selection_callback = None
        self.overlay_mode: bool = False
        self.annotation_drag_callback = None
        self.set_style()

    def set_style(self) -> None:
        pg.setConfigOptions(antialias=True, background="w", foreground="k")

    def set_data(self, df: pd.DataFrame) -> None:
        self.data = df.copy()
        self.refresh_plots()

    def set_channels(self, channels: List[str]) -> None:
        self.channels = channels
        self.refresh_plots()

    def refresh_plots(self) -> None:
        # Detach transient items from existing plots to avoid cross-scene removal warnings
        sel_region_vals = None
        cursor_pos = None
        if self.selection_region:
            try:
                sel_region_vals = self.selection_region.getRegion()
            except Exception:
                sel_region_vals = None
            self.selection_region = None  # recreate per scene
        if self.time_cursor:
            try:
                cursor_pos = self.time_cursor.value()
            except Exception:
                cursor_pos = None
            self.time_cursor = None
        self.annotation_regions.clear()
        self.annotation_clone_regions.clear()
        self.deletion_regions.clear()
        self.widget.clear()
        self.plots = []
        if self.data.empty or not self.channels:
            return
        time = self.data["normalized_time"].values
        if self.overlay_mode:
            p = self.widget.addPlot(row=0, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("left", "Signals")
            p.setLabel("bottom", "Time (s)")
            legend = p.addLegend()
            try:
                legend.setLabelTextColor((255, 255, 255))
            except Exception:
                pass
            for ch in self.channels:
                p.plot(time, self.data[ch].values, pen=pg.mkPen(color=self._color_for_channel(ch), width=1.2), name=ch)
            self._style_axes(p)
            self.plots.append(p)
        else:
            for row, ch in enumerate(self.channels):
                p = self.widget.addPlot(row=row, col=0)
                p.showGrid(x=True, y=True, alpha=0.3)
                p.setLabel("left", ch)
                p.setLabel("bottom", "Time (s)")
                p.plot(time, self.data[ch].values, pen=pg.mkPen(color=self._color_for_channel(ch), width=1.2))
                if row > 0:
                    p.setXLink(self.plots[0])
                self._style_axes(p)
                self.plots.append(p)
        if self.selection_region:
            for p in self.plots:
                p.addItem(self.selection_region)
        if self.time_cursor:
            for p in self.plots:
                p.addItem(self.time_cursor)
        if sel_region_vals:
            self.set_selection(*sel_region_vals)
        if cursor_pos is not None:
            self.set_time_cursor(cursor_pos)

    def _color_for_channel(self, ch: str) -> Tuple[int, int, int]:
        palette = [
            (78, 121, 167),
            (255, 87, 87),
            (89, 161, 79),
            (242, 142, 43),
            (237, 201, 72),
            (144, 103, 167),
            (188, 189, 34),
        ]
        idx = abs(hash(ch)) % len(palette)
        return palette[idx]

    def ensure_selection_region(self) -> None:
        if self.selection_region is None:
            self.selection_region = pg.LinearRegionItem(values=(0, 0.1), movable=True, brush=(200, 200, 255, 80))
            self.selection_region.setZValue(-10)
            if self.selection_callback:
                self.selection_region.sigRegionChanged.connect(self._emit_selection)  # type: ignore
            for p in self.plots:
                p.addItem(self.selection_region)

    def set_selection(self, start: float, end: float) -> None:
        self.ensure_selection_region()
        self.selection_region.setRegion((start, end))

    def clear_selection(self) -> None:
        if self.selection_region:
            for p in self.plots:
                try:
                    if self.selection_region.scene() is p.scene():
                        p.removeItem(self.selection_region)
                except Exception:
                    pass
        self.selection_region = None

    def set_time_cursor(self, t: float) -> None:
        if self.time_cursor is not None:
            try:
                for p in self.plots:
                    if self.time_cursor.scene() is p.scene():
                        p.removeItem(self.time_cursor)
            except Exception:
                pass
        self.time_cursor = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen(self.marker_color, width=1.5))
        for p in self.plots:
            p.addItem(self.time_cursor)

    def update_annotations(self, annotations: List[AnnotationSegment], deletions: List[Tuple[float, float]]) -> None:
        # clear existing
        scene = self.widget.scene()
        for region in self.annotation_regions + self.annotation_clone_regions + self.deletion_regions:
            try:
                if scene is not None:
                    scene.removeItem(region)
            except Exception:
                pass
        # also remove any stray annotation items left on the scene
        if scene is not None:
            for item in list(scene.items()):
                if hasattr(item, "annot_id"):
                    try:
                        scene.removeItem(item)
                    except Exception:
                        pass
        self.annotation_regions.clear()
        self.annotation_clone_regions.clear()
        self.deletion_regions.clear()
        for start, end in deletions:
            reg = pg.LinearRegionItem(values=(start, end), brush=(255, 180, 180, 120), movable=False)
            reg.setZValue(-15)
            self.deletion_regions.append(reg)
            for p in self.plots:
                p.addItem(reg)
        for ann in annotations:
            color = QtGui.QColor(78, 121, 167, 90)
            try:
                color = QtGui.QColor(ann.color)
                color.setAlpha(90)
            except Exception:
                pass
            base_brush = pg.mkBrush(color)
            reg = pg.LinearRegionItem(values=(ann.start, ann.end), brush=base_brush, movable=True)
            reg.setZValue(-12)
            reg.annot_id = getattr(ann, "id", None)
            reg.annot_track = ann.track
            reg._base_brush = base_brush  # type: ignore[attr-defined]
            reg.sigRegionChangeFinished.connect(self._on_region_dragged)  # type: ignore
            self.annotation_regions.append(reg)
            for idx, p in enumerate(self.plots):
                if idx == 0:
                    p.addItem(reg)
                else:
                    clone = pg.LinearRegionItem(values=(ann.start, ann.end), brush=base_brush, movable=False)
                    clone.setZValue(-12)
                    clone.annot_id = getattr(ann, "id", None)
                    clone.annot_track = ann.track
                    clone._base_brush = base_brush  # type: ignore[attr-defined]
                    self.annotation_clone_regions.append(clone)
                    p.addItem(clone)

    def highlight_annotation(self, ann_id: Optional[int]) -> None:
        valid_regions: List[pg.LinearRegionItem] = []
        valid_clones: List[pg.LinearRegionItem] = []
        for reg in self.annotation_regions:
            try:
                base = getattr(reg, "_base_brush", reg.opts.get("brush"))  # type: ignore[attr-defined]
                reg.setBrush(base)
                valid_regions.append(reg)
            except Exception:
                continue
        for reg in self.annotation_clone_regions:
            try:
                base = getattr(reg, "_base_brush", reg.opts.get("brush"))  # type: ignore[attr-defined]
                reg.setBrush(base)
                valid_clones.append(reg)
            except Exception:
                continue
        self.annotation_regions = valid_regions
        self.annotation_clone_regions = valid_clones
        if ann_id is None:
            return
        for reg in self.annotation_regions + self.annotation_clone_regions:
            try:
                if getattr(reg, "annot_id", None) == ann_id:
                    reg.setBrush(pg.mkBrush(QtGui.QColor(255, 200, 0, 120)))
                    break
            except Exception:
                continue

    def set_selection_callback(self, cb) -> None:
        """Register a callback receiving (start, end) when user drags selection."""
        self.selection_callback = cb
        if self.selection_region is not None:
            self.selection_region.sigRegionChanged.connect(self._emit_selection)  # type: ignore

    def _emit_selection(self) -> None:
        if self.selection_callback and self.selection_region:
            start, end = self.selection_region.getRegion()
            self.selection_callback(start, end)

    def _style_axes(self, plot: pg.PlotItem) -> None:
        """Force visible axis lines and readable tick styling."""
        pen = pg.mkPen((80, 80, 80), width=1)
        left = plot.getAxis("left")
        bottom = plot.getAxis("bottom")
        left.setPen(pen)
        left.setTextPen(pen)
        bottom.setPen(pen)
        bottom.setTextPen(pen)
        left.setStyle(tickLength=-5, tickTextOffset=6)
        bottom.setStyle(tickLength=-5, tickTextOffset=6)
        plot.showAxis("bottom", show=True)
        plot.showAxis("left", show=True)

    def set_overlay_mode(self, enabled: bool) -> None:
        self.overlay_mode = enabled
        self.refresh_plots()

    def focus_on(self, start: float, end: float) -> None:
        """Pan to keep the region visible without changing current zoom."""
        if not self.plots or self.data.empty:
            return
        vb = self.plots[0].getViewBox()
        try:
            cur_left, cur_right = vb.viewRange()[0]
        except Exception:
            cur_left, cur_right = 0.0, 1.0
        cur_width = max(cur_right - cur_left, 1e-6)
        lo, hi = sorted((start, end))
        # if region already fully visible, leave zoom unchanged
        if lo >= cur_left and hi <= cur_right:
            return
        mid = (lo + hi) / 2.0
        new_left = mid - cur_width / 2.0
        new_right = mid + cur_width / 2.0
        t_min = float(self.data["normalized_time"].min())
        t_max = float(self.data["normalized_time"].max())
        span = t_max - t_min
        if span > 0:
            new_left = max(t_min, min(new_left, t_max - cur_width))
            new_right = new_left + cur_width
        vb.setXRange(new_left, new_right, padding=0.0)

    def _on_region_dragged(self) -> None:
        """Forward draggable region changes back to callback for live updates."""
        if self.selection_callback is None and self.annotation_drag_callback is None:
            return
        for reg in self.annotation_regions:
            start, end = reg.getRegion()
            for clone in self.annotation_clone_regions:
                if getattr(clone, "annot_id", None) == getattr(reg, "annot_id", None):
                    try:
                        clone.setRegion((start, end))
                    except Exception:
                        pass
            if self.annotation_drag_callback and getattr(reg, "annot_id", None) is not None:
                self.annotation_drag_callback(reg.annot_id, start, end)
            else:
                self.selection_callback(start, end)

    def set_annotation_drag_callback(self, cb) -> None:
        self.annotation_drag_callback = cb

