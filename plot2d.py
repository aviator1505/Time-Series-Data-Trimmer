"""2D plotting controller built on pyqtgraph."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui
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
        self.plot_style: str = "line"
        self.channel_styles: Dict[str, str] = {}
        self.season_period: float = 1.0
        self.marker_color = (80, 80, 220)
        self.selection_callback = None
        self.overlay_mode: bool = False
        self.annotation_drag_callback = None
        self.hover_label = pg.TextItem(
            "",
            anchor=(0, 1),
            color=(30, 30, 30),
            border=pg.mkPen((60, 60, 60)),
            fill=pg.mkBrush(255, 255, 255, 230),
        )
        self.hover_label.setZValue(20)
        self.hover_label.setVisible(False)
        self.hover_dot = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen((40, 40, 40), width=1.2), brush=pg.mkBrush(255, 255, 255), pxMode=True
        )
        self.hover_dot.setZValue(19)
        self.hover_dot.setVisible(False)
        scene = self.widget.scene()
        self._hover_proxy = pg.SignalProxy(scene.sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved) if scene else None
        self._hover_plot: Optional[pg.PlotItem] = None
        self.plot_channels: Dict[pg.PlotItem, List[str]] = {}
        self.time_values: np.ndarray = np.array([])
        self.hover_threshold_px = 20.0
        self.set_style()

    def set_style(self) -> None:
        pg.setConfigOptions(antialias=True, background="w", foreground="k")

    def set_data(self, df: pd.DataFrame) -> None:
        self.data = df.copy()
        self.refresh_plots()

    def set_channels(self, channels: List[str]) -> None:
        # keep only channels that exist in current data (if loaded)
        if not self.data.empty:
            self.channels = [ch for ch in channels if ch in self.data.columns]
        else:
            self.channels = channels
        # prune styles for removed channels
        self.channel_styles = {ch: style for ch, style in self.channel_styles.items() if ch in self.channels}
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
        self._reset_hover_items()
        self.annotation_regions.clear()
        self.annotation_clone_regions.clear()
        self.deletion_regions.clear()
        self.widget.clear()
        self.plots = []
        self.plot_channels = {}
        self.time_values = np.array([])
        if self.data.empty or not self.channels:
            return
        time = self.data["normalized_time"].values
        if self.plot_style == "heatmap":
            self._draw_heatmap(time)
        elif self.plot_style == "seasonal":
            self._draw_seasonal(time)
        else:
            self._draw_standard(time)
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

    def _reset_hover_items(self) -> None:
        """Remove hover helpers from any previous plot before rebuilding."""
        if self._hover_plot:
            try:
                self._hover_plot.removeItem(self.hover_label)
                self._hover_plot.removeItem(self.hover_dot)
            except Exception:
                pass
        self._hover_plot = None
        self.hover_label.setVisible(False)
        self.hover_dot.setVisible(False)

    def _hide_hover(self) -> None:
        self.hover_label.setVisible(False)
        self.hover_dot.setVisible(False)

    def _show_hover(self, plot: pg.PlotItem, t_val: float, y_val: float, ch: str) -> None:
        if self._hover_plot is not plot:
            if self._hover_plot:
                try:
                    self._hover_plot.removeItem(self.hover_label)
                    self._hover_plot.removeItem(self.hover_dot)
                except Exception:
                    pass
            self._hover_plot = plot
            plot.addItem(self.hover_label)
            plot.addItem(self.hover_dot)
        self.hover_label.setText(f"{ch}: t={t_val:.3f}s, y={y_val:.3f}")
        self.hover_label.setPos(t_val, y_val)
        self.hover_label.setVisible(True)
        self.hover_dot.setData([t_val], [y_val])
        self.hover_dot.setVisible(True)

    def _nearest_point(self, plot: pg.PlotItem, scene_pos: QtCore.QPointF) -> Optional[Tuple[str, float, float]]:
        channels = self.plot_channels.get(plot, [])
        if not channels or self.time_values.size == 0:
            return None
        vb = plot.getViewBox()
        if vb is None:
            return None
        view_pos = vb.mapSceneToView(scene_pos)
        t_guess = view_pos.x()
        idx = int(np.searchsorted(self.time_values, t_guess))
        candidate_idxs = [idx - 1, idx, idx + 1]
        candidate_idxs = [i for i in candidate_idxs if 0 <= i < len(self.time_values)]
        best: Optional[Tuple[str, float, float]] = None
        best_dist: Optional[float] = None
        for i in candidate_idxs:
            t_val = float(self.time_values[i])
            for ch in channels:
                if ch not in self.data.columns:
                    continue
                y_val = float(self.data[ch].iat[i])
                if pd.isna(y_val):
                    continue
                pt_scene = vb.mapViewToScene(QtCore.QPointF(t_val, y_val))
                dx = float(pt_scene.x() - scene_pos.x())
                dy = float(pt_scene.y() - scene_pos.y())
                dist = dx * dx + dy * dy
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best = (ch, t_val, y_val)
        if best_dist is None or best_dist > self.hover_threshold_px ** 2:
            return None
        return best

    def _on_mouse_moved(self, evt) -> None:
        if not self.plots or self.data.empty:
            self._hide_hover()
            return
        try:
            scene_pos = evt[0]
        except Exception:
            return
        for plot in self.plots:
            vb = plot.getViewBox()
            if vb is None:
                continue
            try:
                if not vb.sceneBoundingRect().contains(scene_pos):
                    continue
            except Exception:
                continue
            hit = self._nearest_point(plot, scene_pos)
            if hit is None:
                self._hide_hover()
                return
            ch, t_val, y_val = hit
            self._show_hover(plot, t_val, y_val, ch)
            return
        self._hide_hover()

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

    # ------------------------------------------------------------------
    # Plot style helpers
    def set_plot_style(self, style: str) -> None:
        allowed = {"line", "scatter", "area", "seasonal", "heatmap"}
        if style not in allowed:
            style = "line"
        self.plot_style = style
        self.refresh_plots()

    def set_channel_style(self, channel: str, style: Optional[str]) -> None:
        allowed = {"line", "scatter", "area"}
        if style is None or style == "" or style not in allowed:
            if channel in self.channel_styles:
                self.channel_styles.pop(channel, None)
                self.refresh_plots()
            return
        self.channel_styles[channel] = style
        self.refresh_plots()

    def set_season_period(self, period: float) -> None:
        if period <= 0:
            return
        self.season_period = period
        if self.plot_style == "seasonal":
            self.refresh_plots()

    def _draw_standard(self, time: np.ndarray) -> None:
        self.time_values = time
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
                if ch not in self.data.columns:
                    continue
                self._plot_series(p, time, self.data[ch].values, ch, name=ch)
            self._style_axes(p)
            self.plots.append(p)
            self.plot_channels[p] = [ch for ch in self.channels if ch in self.data.columns]
        else:
            for row, ch in enumerate(self.channels):
                if ch not in self.data.columns:
                    continue
                p = self.widget.addPlot(row=row, col=0)
                p.showGrid(x=True, y=True, alpha=0.3)
                p.setLabel("left", ch)
                p.setLabel("bottom", "Time (s)")
                self._plot_series(p, time, self.data[ch].values, ch)
                if row > 0:
                    p.setXLink(self.plots[0])
                self._style_axes(p)
                self.plots.append(p)
                self.plot_channels[p] = [ch]

    def _plot_series(self, plot: pg.PlotItem, time: np.ndarray, values: np.ndarray, ch: str, name: Optional[str] = None) -> None:
        color = self._color_for_channel(ch)
        style = self.channel_styles.get(ch, self.plot_style)
        if style == "scatter":
            plot.plot(
                time,
                values,
                pen=None,
                symbol="o",
                symbolSize=6,
                symbolPen=pg.mkPen(color=color, width=0.8),
                symbolBrush=pg.mkBrush(color),
                name=name,
            )
        elif style == "area":
            plot.plot(
                time,
                values,
                pen=pg.mkPen(color=color, width=1.2),
                brush=pg.mkBrush(color[0], color[1], color[2], 80),
                fillLevel=0,
                name=name,
            )
        else:  # line
            plot.plot(time, values, pen=pg.mkPen(color=color, width=1.2), name=name)

    def _draw_seasonal(self, time: np.ndarray) -> None:
        base = time - float(time.min())
        period = max(self.season_period, 1e-9)
        season_ids = np.floor(base / period).astype(int)
        t_mod = np.mod(base, period)
        unique_seasons = np.unique(season_ids)
        # disable hover in seasonal view to avoid ambiguous nearest points
        self.time_values = np.array([])
        if self.overlay_mode:
            p = self.widget.addPlot(row=0, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("left", "Signals")
            p.setLabel("bottom", "Time within period (s)")
            legend = p.addLegend()
            for ch in self.channels:
                if ch not in self.data.columns:
                    continue
                vals = self.data[ch].values
                color = self._color_for_channel(ch)
                for idx, sid in enumerate(unique_seasons):
                    mask = season_ids == sid
                    if np.count_nonzero(mask) < 2:
                        continue
                    p.plot(
                        t_mod[mask],
                        vals[mask],
                        pen=pg.mkPen((color[0], color[1], color[2], 180), width=1.0),
                        name=ch if idx == 0 else None,
                    )
            self._style_axes(p)
            self.plots.append(p)
            self.plot_channels[p] = []
        else:
            for row, ch in enumerate(self.channels):
                if ch not in self.data.columns:
                    continue
                p = self.widget.addPlot(row=row, col=0)
                p.showGrid(x=True, y=True, alpha=0.3)
                p.setLabel("left", ch)
                p.setLabel("bottom", "Time within period (s)")
                vals = self.data[ch].values
                color = self._color_for_channel(ch)
                for sid in unique_seasons:
                    mask = season_ids == sid
                    if np.count_nonzero(mask) < 2:
                        continue
                    p.plot(
                        t_mod[mask],
                        vals[mask],
                        pen=pg.mkPen((color[0], color[1], color[2], 180), width=1.0),
                    )
                if row > 0 and self.plots:
                    p.setXLink(self.plots[0])
                self._style_axes(p)
                self.plots.append(p)
                self.plot_channels[p] = []

    def _draw_heatmap(self, time: np.ndarray) -> None:
        # disable hover for heatmap view
        self.time_values = np.array([])
        p = self.widget.addPlot(row=0, col=0)
        p.showGrid(x=True, y=True, alpha=0.15)
        p.setLabel("left", "Channels")
        p.setLabel("bottom", "Time (s)")
        matrices: List[np.ndarray] = []
        ticks: List[Tuple[float, str]] = []
        for idx, ch in enumerate(self.channels):
            if ch not in self.data.columns:
                continue
            matrices.append(self.data[ch].to_numpy())
            ticks.append((idx + 0.5, ch))
        if not matrices:
            self.plots.append(p)
            self.plot_channels[p] = []
            return
        img_array = np.vstack(matrices)
        img_array = np.nan_to_num(img_array, nan=0.0)
        span = float(time.max() - time.min()) if len(time) else 1.0
        span = max(span, 1e-6)
        img_item = pg.ImageItem(img_array)
        img_item.setRect(QtCore.QRectF(float(time.min()), 0.0, span, len(matrices)))
        p.addItem(img_item)
        left_axis = p.getAxis("left")
        left_axis.setTicks([ticks, []])
        p.setYRange(0, len(matrices))
        self._style_axes(p)
        self.plots.append(p)
        self.plot_channels[p] = []

