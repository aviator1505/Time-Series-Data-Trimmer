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
        self.scatter = gl.GLScatterPlotItem()
        self.view.addItem(self.scatter)

    def set_data(self, df: pd.DataFrame) -> None:
        self.data = df.copy()

    def set_mappings(self, mappings: Dict[str, Dict[str, str]]) -> None:
        """Mappings of body part -> {x,y,z} column names."""
        self.mappings = mappings

    def update_time(self, t: float) -> None:
        if self.data.empty:
            return
        idx = (self.data["normalized_time"] - t).abs().idxmin()
        row = self.data.loc[idx]
        points = []
        colors = []
        # try to build from mappings, else derive from heading angles to simple circle positions
        for part, cols in self.mappings.items():
            xcol = cols.get("x")
            ycol = cols.get("y")
            zcol = cols.get("z")
            try:
                x = float(row[xcol]) if xcol in row else 0.0
                y = float(row[ycol]) if ycol in row else 0.0
                z = float(row[zcol]) if zcol in row else 0.0
            except Exception:
                x = y = z = 0.0
            points.append([x, y, z])
            colors.append((0.3, 0.3, 0.8, 1.0))
        if not points:
            # fallback using heading angles into a star layout
            for idx, name in enumerate(["head_heading_deg", "torso_heading_deg", "chair_heading_deg", "left_foot_heading_deg", "right_foot_heading_deg"]):
                if name not in self.data.columns:
                    continue
                heading = math.radians(float(row[name]))
                radius = 1.0 + idx * 0.2
                x = radius * math.cos(heading)
                y = radius * math.sin(heading)
                z = idx * 0.05
                points.append([x, y, z])
                colors.append((0.2 + 0.1 * idx, 0.1, 0.4 + 0.1 * idx, 1.0))
        if not points:
            return
        pts = np.array(points)
        self.scatter.setData(pos=pts, color=np.array(colors))

