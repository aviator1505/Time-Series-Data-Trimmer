"""3D plotting controller using pyqtgraph.opengl."""
from __future__ import annotations

import math
from typing import Dict, Optional, Iterable

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
        self.axes: Dict[str, Dict[str, gl.GLLinePlotItem]] = {}
        self.skeleton: Dict[tuple[str, str], gl.GLLinePlotItem] = {}
        self.trails: Dict[str, gl.GLLinePlotItem] = {}
        self.trail_history: Dict[str, list[np.ndarray]] = {}
        self.trail_length = 60  # frames to keep in trail

    def set_data(self, df: pd.DataFrame) -> None:
        self.data = df.copy()
        if not self.mappings:
            self._infer_mappings_from_columns(df.columns)

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
        used_axes = set()
        pos_dict: Dict[str, np.ndarray] = {}
        # try to build from mappings, else derive automatically from column names
        target_parts = list(self.mappings.keys()) if self.mappings else [
            "head",
            "torso",
            "chair",
            "left_foot",
            "right_foot",
            "workspace",
            "screen",
        ]
        for idx_part, part in enumerate(target_parts):
            pos, rot, has_translation = self._extract_pose(part, row, idx_part, len(target_parts))
            if pos is None:
                continue
            points.append(pos.tolist())
            colors.append((0.3, 0.3, 0.8, 1.0))
            self._draw_axes(part, pos, rot)
            used_axes.add(part)
            heading = self._heading_from_rotation(part, rot, row)
            self._draw_arrow(part, *pos, heading, long_arrow=not has_translation)
            self._place_label(part, *pos)
            used_labels.add(part)
            pos_dict[part] = pos
        if not points:
            # fallback using heading angles into a star layout
            names = ["head_heading_deg", "torso_heading_deg", "chair_heading_deg", "left_foot_heading_deg", "right_foot_heading_deg"]
            for idx, name in enumerate(names):
                if name not in self.data.columns:
                    continue
                x, y, z = self._anchor_position(idx, len(names))
                points.append([x, y, z])
                colors.append((0.2 + 0.1 * idx, 0.1, 0.4 + 0.1 * idx, 1.0))
                self._draw_axes(name, np.array([x, y, z]), np.eye(3))
                self._draw_arrow(name, x, y, z, float(row.get(name, 0.0)), long_arrow=True)
                self._place_label(name, x, y, z)
                used_labels.add(name)
                used_axes.add(name)
                pos_dict[name] = np.array([x, y, z], dtype=float)
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
        self._cleanup_axes(used_axes)
        self._update_skeleton(pos_dict)
        self._update_trails(pos_dict)

    def _heading_from_rotation(self, part: str, rot: np.ndarray, row: pd.Series) -> float:
        """Derive heading from rotation matrix or fallbacks."""
        try:
            fwd = rot[2]  # forward row
            return math.degrees(math.atan2(fwd[1], fwd[0]))
        except Exception:
            pass
        return float(row.get(f"{part}_heading_deg", 0.0))

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

    def _yaw_from_quat(self, x: float, y: float, z: float, w: float) -> float:
        """Return yaw (heading) from quaternion in radians."""
        # yaw (Z) from quaternion
        t0 = +2.0 * (w * z + x * y)
        t1 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t0, t1)

    def _rotation_matrix(self, part: str, row: pd.Series, mapping: Dict[str, str]) -> np.ndarray:
        """Return 3x3 rotation matrix from quaternion, euler, or direction vectors; identity if unavailable."""
        qx = mapping.get("qx")
        qy = mapping.get("qy")
        qz = mapping.get("qz")
        qw = mapping.get("qw")
        if all(k in mapping for k in ["qx", "qy", "qz", "qw"]) and all(col in row for col in (qx, qy, qz, qw)):
            try:
                return self._quat_to_mat(float(row[qw]), float(row[qx]), float(row[qy]), float(row[qz]))
            except Exception:
                pass
        # Euler yaw/pitch/roll (degrees)
        yaw_key = mapping.get("yaw") or mapping.get("yaw_deg") or f"{part}_yaw_deg"
        pitch_key = mapping.get("pitch") or mapping.get("pitch_deg") or f"{part}_pitch_deg"
        roll_key = mapping.get("roll") or mapping.get("roll_deg") or f"{part}_roll_deg"
        if all(k in row for k in (yaw_key, pitch_key, roll_key)):
            try:
                return self._euler_to_mat(
                    math.radians(float(row[yaw_key])),
                    math.radians(float(row[pitch_key])),
                    math.radians(float(row[roll_key])),
                )
            except Exception:
                pass
        dx = mapping.get("dx")
        dy = mapping.get("dy")
        dz = mapping.get("dz")
        if dx and dy and all(c in row for c in (dx, dy)):
            try:
                fwd = np.array([float(row[dx]), float(row[dy]), float(row.get(dz, 0.0))], dtype=float)
                fwd_norm = fwd / (np.linalg.norm(fwd) + 1e-9)
                # construct simple frame with world up
                up = np.array([0.0, 0.0, 1.0])
                right = np.cross(up, fwd_norm)
                right /= (np.linalg.norm(right) + 1e-9)
                up = np.cross(fwd_norm, right)
                return np.vstack([right, up, fwd_norm])
            except Exception:
                pass
        return np.eye(3)

    def _euler_to_mat(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """Z (yaw), Y (pitch), X (roll) intrinsic rotation."""
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=float,
        )

    def _quat_to_mat(self, w: float, x: float, y: float, z: float) -> np.ndarray:
        """Quaternion to rotation matrix."""
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (xx + zz), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (xx + yy)],
            ],
            dtype=float,
        )

    def _draw_axes(self, part: str, origin: np.ndarray, rot: np.ndarray, length: float = 0.25) -> None:
        """Draw orientation triad for a part."""
        # remove old
        for axis in ["x", "y", "z"]:
            if part in self.axes and axis in self.axes[part]:
                try:
                    self.view.removeItem(self.axes[part][axis])
                except Exception:
                    pass
        basis = {
            "x": (np.array([1, 0, 0]), (1.0, 0.2, 0.2, 1.0)),
            "y": (np.array([0, 1, 0]), (0.2, 1.0, 0.2, 1.0)),
            "z": (np.array([0, 0, 1]), (0.2, 0.4, 1.0, 1.0)),
        }
        self.axes.setdefault(part, {})
        for axis, (vec, color) in basis.items():
            end = origin + rot @ vec * length
            line_pts = np.vstack([origin, end])
            item = gl.GLLinePlotItem(pos=line_pts, color=np.array(color), width=2, antialias=True)
            self.axes[part][axis] = item
            self.view.addItem(item)

    def _cleanup_axes(self, keep: set) -> None:
        for part in list(self.axes.keys()):
            if part not in keep:
                for axis_item in self.axes[part].values():
                    try:
                        self.view.removeItem(axis_item)
                    except Exception:
                        pass
                self.axes.pop(part, None)

    # ------------------------------------------------------------------
    # Skeleton & trails
    # ------------------------------------------------------------------
    def _update_skeleton(self, positions: Dict[str, np.ndarray]) -> None:
        """Connect key parts with bone segments when positions exist."""
        connections = [
            ("head", "torso"),
            ("torso", "left_foot"),
            ("torso", "right_foot"),
            ("torso", "chair"),
            ("head", "chair"),
        ]
        for bone in connections:
            if bone[0] not in positions or bone[1] not in positions:
                continue
            pts = np.vstack([positions[bone[0]], positions[bone[1]]])
            if bone in self.skeleton:
                try:
                    self.skeleton[bone].setData(pos=pts)
                except Exception:
                    pass
            else:
                item = gl.GLLinePlotItem(pos=pts, color=np.array([0.7, 0.7, 0.7, 0.6]), width=2, antialias=True)
                self.skeleton[bone] = item
                self.view.addItem(item)

    def _update_trails(self, positions: Dict[str, np.ndarray]) -> None:
        """Draw motion trails for each part."""
        for part, pos in positions.items():
            hist = self.trail_history.setdefault(part, [])
            hist.append(pos)
            if len(hist) > self.trail_length:
                hist.pop(0)
            line = np.vstack(hist)
            if part in self.trails:
                try:
                    self.trails[part].setData(pos=line)
                except Exception:
                    pass
            else:
                color = np.array([0.5, 0.7, 1.0, 0.4])
                item = gl.GLLinePlotItem(pos=line, color=color, width=1.5, antialias=True)
                self.trails[part] = item
                self.view.addItem(item)

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

    # ------------------------------------------------------------------
    # Mapping inference helpers
    # ------------------------------------------------------------------
    def _infer_mappings_from_columns(self, columns: Iterable[str]) -> None:
        """Infer position/orientation columns based on name patterns."""
        cols = list(columns)
        parts = {
            "head": ["head", "mocap_head"],
            "torso": ["torso", "chest"],
            "chair": ["chair"],
            "left_foot": ["leftfoot", "left_foot"],
            "right_foot": ["rightfoot", "right_foot"],
            "workspace": ["workspace"],
            "screen": ["screen"],
        }
        inferred: Dict[str, Dict[str, str]] = {}
        for key, aliases in parts.items():
            entry = self._auto_components_for_part(key, aliases, cols)
            if entry:
                inferred[key] = entry
        if inferred:
            self.mappings = inferred

    def _normalize(self, name: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in name)

    def _find_best_base(self, columns: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
        norm_cols = {self._normalize(c): c for c in columns}
        for alias in aliases:
            for norm, orig in norm_cols.items():
                if alias.replace("_", "") in norm.replace("_", ""):
                    return alias
        return None

    def _match_component(self, columns: Iterable[str], base: str, comp: str) -> Optional[str]:
        base_clean = base.replace("_", "")
        candidates = []
        for col in columns:
            norm = self._normalize(col)
            if base_clean in norm.replace("_", "") and norm.endswith(comp):
                candidates.append(col)
        if candidates:
            # pick shortest name to avoid overly specific variants
            return sorted(candidates, key=len)[0]
        return None

    def _auto_components_for_part(self, part: str, aliases: Iterable[str], columns: Iterable[str]) -> Dict[str, str]:
        """Return component mapping for a part by scanning available columns."""
        entry: Dict[str, str] = {}
        for comp in ["x", "y", "z", "qx", "qy", "qz", "qw", "dx", "dy", "dz", "yaw", "pitch", "roll"]:
            found = None
            for alias in aliases:
                found = self._match_component(columns, alias, comp)
                if found:
                    break
            if found:
                entry[comp] = found
        return entry

    def _extract_pose(self, part: str, row: pd.Series, idx: int, total: int) -> tuple[Optional[np.ndarray], np.ndarray, bool]:
        """Get position and rotation for part; position None if unavailable."""
        mapping = self.mappings.get(part, {})
        if not mapping:
            mapping = self._auto_components_for_part(part, self._part_aliases(part), row.index)
        pos = None
        has_translation = False
        if all(k in mapping and mapping[k] in row for k in ["x", "y", "z"]):
            try:
                pos = np.array([float(row[mapping["x"]]), float(row[mapping["y"]]), float(row[mapping["z"]])], dtype=float)
                has_translation = True
            except Exception:
                pos = None
        if pos is None:
            pos = np.array(self._anchor_position(idx, total), dtype=float)
        rot = self._rotation_matrix(part, row, mapping)
        return pos, rot, has_translation

    def _part_aliases(self, part: str) -> Iterable[str]:
        """Common aliases per body part."""
        mapping = {
            "head": ["head", "mocap_head"],
            "torso": ["torso", "chest", "spine"],
            "chair": ["chair", "seat"],
            "left_foot": ["left_foot", "lf", "leftfoot"],
            "right_foot": ["right_foot", "rf", "rightfoot"],
            "workspace": ["workspace"],
            "screen": ["screen", "display"],
        }
        return mapping.get(part, [part])

