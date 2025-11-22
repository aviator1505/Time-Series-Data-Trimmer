"""3D plotting controller using pyqtgraph.opengl."""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyqtgraph.opengl as gl
from PyQt6 import QtGui


class PlotController3D:
    def __init__(self) -> None:
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(QtGui.QColor(20, 20, 20))
        self.view.opts["distance"] = 6
        self.grid = gl.GLGridItem()
        self.view.addItem(self.grid)
        self.data: pd.DataFrame = pd.DataFrame()
        self.mappings: Dict[str, Dict[str, str]] = {}
        self.frames: Dict[str, Dict] = {}
        self.scatter = gl.GLScatterPlotItem()
        self.view.addItem(self.scatter)
        self.arrows: Dict[str, gl.GLLinePlotItem] = {}
        self.labels: Dict[str, gl.GLTextItem] = {}
        self.active_channels: Dict[str, str] = {}

    def set_data(self, df: pd.DataFrame) -> None:
        self.data = df.copy()

    def set_mappings(self, mappings: Dict[str, Dict[str, str]]) -> None:
        """Mappings of body part -> {x,y,z} column names."""
        self.mappings = mappings

    def set_frames(self, frames: Dict[str, Dict]) -> None:
        """Store coordinate frame offsets (simple heading offsets only)."""
        self.frames = frames

    def set_active_channels(self, channels: Dict[str, str] | Dict[str, None] | Dict[str, str]) -> None:
        """Keep track of channels currently visible in 2D so we mirror them in 3D."""
        # store as dict for potential future metadata; value unused for now
        self.active_channels = {ch: "" for ch in channels}

    def update_time(self, t: float) -> None:
        if self.data.empty:
            return
        idx = (self.data["normalized_time"] - t).abs().idxmin()
        row = self.data.loc[idx]
        points = []
        colors = []
        used_labels = set()
        # try to build from mappings, else derive from heading angles to simple circle positions
        mapped_items = list(self.mappings.items())
        for idx_part, (part, cols) in enumerate(mapped_items):
            xcol = cols.get("x")
            ycol = cols.get("y")
            zcol = cols.get("z")
            translational = any(c in row for c in (xcol, ycol, zcol) if c)
            if translational:
                try:
                    x = float(row[xcol]) if xcol in row else 0.0
                    y = float(row[ycol]) if ycol in row else 0.0
                    z = float(row[zcol]) if zcol in row else 0.0
                except Exception:
                    x = y = z = 0.0
            else:
                x, y, z = self._anchor_position(idx_part, len(mapped_items))
            points.append([x, y, z])
            colors.append((0.3, 0.3, 0.8, 1.0))
            self._draw_arrow(part, x, y, z, self._heading_for_part(part, row), long_arrow=not translational)
            self._place_label(part, x, y, z)
            used_labels.add(part)
        if not points:
            # fallback using heading angles into a star layout
            names = ["head_heading_deg", "torso_heading_deg", "chair_heading_deg", "left_foot_heading_deg", "right_foot_heading_deg"]
            for idx, name in enumerate(names):
                if name not in self.data.columns:
                    continue
                x, y, z = self._anchor_position(idx, len(names))
                points.append([x, y, z])
                colors.append((0.2 + 0.1 * idx, 0.1, 0.4 + 0.1 * idx, 1.0))
                self._draw_arrow(name, x, y, z, float(row.get(name, 0.0)), long_arrow=True)
                self._place_label(name, x, y, z)
                used_labels.add(name)
        # if active channels are set, mirror them as small columns around a circle
        if self.active_channels and self.data is not None and not self.data.empty:
            chans = list(self.active_channels.keys())
            radius = 1.2
            for idx, ch in enumerate(chans):
                if ch not in row:
                    continue
                angle = 2 * math.pi * idx / max(len(chans), 1)
                val = float(row[ch])
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = val * 0.01  # small vertical displacement for differentiation
                points.append([x, y, z])
                colors.append((0.6, 0.8, 0.2, 0.9))
                self._place_label(ch, x, y, z)
                used_labels.add(ch)
        if not points:
            return
        pts = np.array(points)
        self.scatter.setData(pos=pts, color=np.array(colors))
        self._cleanup_labels(used_labels)

    def _heading_for_part(self, part: str, row: pd.Series) -> float:
        key = f"{part}_heading_deg"
        return float(row.get(key, 0.0))

    def _draw_arrow(self, part: str, x: float, y: float, z: float, heading_deg: float, long_arrow: bool = False) -> None:
        # remove existing
        if part in self.arrows:
            try:
                self.view.removeItem(self.arrows[part])
            except Exception:
                pass
        length = 0.8 if long_arrow else 0.4
        angle_rad = math.radians(heading_deg + self._frame_offset(part))
        end = np.array([x + length * math.cos(angle_rad), y + length * math.sin(angle_rad), z])
        line_pts = np.vstack([[x, y, z], end])
        color = np.array([0.9, 0.6, 0.2, 1.0])
        arrow = gl.GLLinePlotItem(pos=line_pts, color=color, width=3, antialias=True)
        self.arrows[part] = arrow
        self.view.addItem(arrow)

    def _frame_offset(self, part: str) -> float:
        # simple heading offset based on frames dict: if part frame exists, use offset
        info = self.frames.get(part, {})
        return float(info.get("offset", 0.0))

    def _anchor_position(self, idx: int, total: int) -> tuple[float, float, float]:
        """Static anchor positions when no translation data exists."""
        radius = 0.8
        if total > 0:
            angle = 2 * math.pi * idx / max(total, 1)
        else:
            angle = 0.0
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 0.05 * idx
        return x, y, z

    def _place_label(self, part: str, x: float, y: float, z: float) -> None:
        """Attach/update a text label near the marker."""
        try:
            pos = np.array([x, y, z], dtype=float)
            label = self.labels.get(part)
            if label is None:
                label = gl.GLTextItem(
                    text=part,
                    pos=pos,
                    color=QtGui.QColor(240, 240, 240),
                    font=QtGui.QFont("Helvetica", 10),
                )
                self.labels[part] = label
                self.view.addItem(label)
            else:
                label.setData(pos=pos, text=part)
        except Exception:
            # silently ignore labeling failures to keep rendering robust
            pass

    def _cleanup_labels(self, keep: set) -> None:
        for part in list(self.labels.keys()):
            if part not in keep:
                try:
                    self.view.removeItem(self.labels[part])
                except Exception:
                    pass
                self.labels.pop(part, None)

