"""3D plotting controller using pyqtgraph.opengl."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Iterable

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
        # Track parts using fallback positions for visual indication (Issue 2 fix)
        self.fallback_parts: set = set()
        # Status callback for reporting fallback usage (optional)
        self.status_callback: Optional[callable] = None
        # World and user-defined frame axes visualization
        self.world_axes: Dict[str, gl.GLLinePlotItem] = {}
        self.world_axis_labels: List[gl.GLTextItem] = []
        self.world_label: Optional[gl.GLTextItem] = None
        self.show_world_axes: bool = True  # Toggle for world axes visibility
        self.frame_axes: Dict[str, Dict[str, gl.GLLinePlotItem]] = {}
        self.frame_labels: Dict[str, gl.GLTextItem] = {}
        self.frame_highlight: List[gl.GLLinePlotItem] = []  # Highlighted frame axes
        self.highlighted_frame: Optional[str] = None
        self.show_frame_axes: bool = True  # Toggle for frame axes visibility
        self._draw_world_axes()

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
        # Clear fallback tracking for this frame (Issue 2 fix)
        self.fallback_parts.clear()
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
            # Issue 2 fix: Use orange/yellow color for fallback positions to indicate data issue
            if has_translation:
                colors.append((0.3, 0.3, 0.8, 1.0))  # Normal blue color
            else:
                colors.append((1.0, 0.6, 0.1, 1.0))  # Orange color for fallback
                self.fallback_parts.add(part)
            self._draw_axes(part, pos, rot)
            used_axes.add(part)
            heading = self._heading_from_rotation(part, rot, row)
            # Issue 3 fix: Only draw arrow if heading is valid (not None/NaN)
            if heading is not None and not (isinstance(heading, float) and np.isnan(heading)):
                self._draw_arrow(part, *pos, heading, long_arrow=not has_translation)
            else:
                # Clean up existing arrow when heading becomes unavailable
                self._remove_arrow(part)
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

        # Draw coordinate frame axes at body part positions
        self._update_frame_axes(pos_dict)

        # Issue 2 fix: Emit status message when parts are using fallback positions
        if self.fallback_parts and self.status_callback:
            parts_list = ", ".join(sorted(self.fallback_parts))
            self.status_callback(f"3D fallback positions for: {parts_list}")

    def set_status_callback(self, callback: callable) -> None:
        """Set callback for status messages (e.g., fallback position warnings)."""
        self.status_callback = callback

    # ------------------------------------------------------------------
    # Coordinate Frame Visualization
    # ------------------------------------------------------------------
    def _draw_world_axes(self) -> None:
        """Draw RGB axes at origin showing world coordinate frame.

        Convention: Red=X (forward), Green=Y (left), Blue=Z (up)
        This helps users understand the coordinate system.
        """
        axis_length = 1.0
        axis_width = 3

        # X axis - Red
        x_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [axis_length, 0, 0]], dtype=float),
            color=(1, 0, 0, 1),
            width=axis_width,
            antialias=True
        )
        self.view.addItem(x_axis)
        self.world_axes["X"] = x_axis

        # Y axis - Green
        y_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, axis_length, 0]], dtype=float),
            color=(0, 1, 0, 1),
            width=axis_width,
            antialias=True
        )
        self.view.addItem(y_axis)
        self.world_axes["Y"] = y_axis

        # Z axis - Blue
        z_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, axis_length]], dtype=float),
            color=(0, 0, 1, 1),
            width=axis_width,
            antialias=True
        )
        self.view.addItem(z_axis)
        self.world_axes["Z"] = z_axis

        # Add X/Y/Z labels at axis tips
        self._add_axis_label("X", axis_length + 0.1, 0, 0, (255, 100, 100))
        self._add_axis_label("Y", 0, axis_length + 0.1, 0, (100, 255, 100))
        self._add_axis_label("Z", 0, 0, axis_length + 0.1, (100, 100, 255))

    def _add_axis_label(self, text: str, x: float, y: float, z: float,
                        color: tuple) -> None:
        """Add a text label for an axis at the specified position.

        Args:
            text: Label text (e.g., "X", "Y", "Z").
            x: X coordinate for label position.
            y: Y coordinate for label position.
            z: Z coordinate for label position.
            color: RGB tuple (0-255) for label color.
        """
        try:
            label = gl.GLTextItem(
                text=text,
                pos=np.array([x, y, z], dtype=float),
                color=QtGui.QColor(*color),
                font=QtGui.QFont("Helvetica", 10),
            )
            self.world_axis_labels.append(label)
            self.view.addItem(label)
        except Exception:
            # GLTextItem may not be available in all pyqtgraph versions
            pass

    def set_world_axes_visible(self, visible: bool) -> None:
        """Show or hide the world coordinate frame axes.

        Args:
            visible: True to show world axes, False to hide them.
        """
        self.show_world_axes = visible

        # Toggle visibility of axis lines
        for axis in self.world_axes.values():
            try:
                axis.setVisible(visible)
            except Exception:
                pass

        # Toggle visibility of axis labels
        for label in self.world_axis_labels:
            try:
                label.setVisible(visible)
            except Exception:
                pass

    def draw_frame_at_position(self, frame_name: str, position: np.ndarray, heading_offset: float) -> None:
        """Draw coordinate frame axes at a specific position with heading rotation.

        Args:
            frame_name: Name of the frame (for labeling and tracking).
            position: (x, y, z) position to draw the frame.
            heading_offset: Total heading offset in degrees (from hierarchical computation).
        """
        if not self.show_frame_axes:
            return

        # Remove existing frame axes if present
        if frame_name in self.frame_axes:
            for axis_line in self.frame_axes[frame_name].values():
                try:
                    self.view.removeItem(axis_line)
                except Exception:
                    pass

        # Remove existing label if present
        if frame_name in self.frame_labels:
            try:
                self.view.removeItem(self.frame_labels[frame_name])
            except Exception:
                pass

        axis_length = 0.3
        angle_rad = math.radians(heading_offset)

        # Rotate X and Y axes by heading offset (Z stays vertical)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        # Local X axis (forward direction after rotation)
        x_dir = np.array([cos_a, sin_a, 0])
        # Local Y axis (left direction after rotation)
        y_dir = np.array([-sin_a, cos_a, 0])
        # Z axis stays up
        z_dir = np.array([0, 0, 1])

        # Use brighter colors for highlighted frame
        is_highlighted = (frame_name == self.highlighted_frame)
        if is_highlighted:
            axes_config = [
                ("X", x_dir, (1.0, 0.5, 0.5, 1.0), 3),  # Bright red, thicker
                ("Y", y_dir, (0.5, 1.0, 0.5, 1.0), 3),  # Bright green, thicker
                ("Z", z_dir, (0.5, 0.5, 1.0, 1.0), 3),  # Bright blue, thicker
            ]
        else:
            axes_config = [
                ("X", x_dir, (1.0, 0.4, 0.4, 0.8), 2),  # Light red
                ("Y", y_dir, (0.4, 1.0, 0.4, 0.8), 2),  # Light green
                ("Z", z_dir, (0.4, 0.4, 1.0, 0.8), 2),  # Light blue
            ]

        self.frame_axes[frame_name] = {}
        for axis_name, direction, color, width in axes_config:
            end = position + direction * axis_length
            line_pts = np.vstack([position, end])
            line = gl.GLLinePlotItem(pos=line_pts, color=np.array(color), width=width, antialias=True)
            self.frame_axes[frame_name][axis_name] = line
            self.view.addItem(line)

        # Add frame label slightly offset
        try:
            label_color = QtGui.QColor(255, 255, 200) if is_highlighted else QtGui.QColor(200, 200, 200)
            font_size = 10 if is_highlighted else 8
            label = gl.GLTextItem(
                text=frame_name,
                pos=position + np.array([0, 0, axis_length + 0.05]),
                color=label_color,
                font=QtGui.QFont("Helvetica", font_size),
            )
            self.frame_labels[frame_name] = label
            self.view.addItem(label)
        except Exception:
            pass

    def highlight_frame(self, frame_name: Optional[str]) -> None:
        """Highlight a specific frame's axes (thicker lines, brighter colors).

        Args:
            frame_name: Name of the frame to highlight, or None to clear highlighting.
        """
        self.highlighted_frame = frame_name
        # Redraw will apply highlighting during next update_time()

    def set_frame_axes_visible(self, visible: bool) -> None:
        """Toggle visibility of all user-defined frame axes.

        Args:
            visible: True to show frame axes, False to hide them.
        """
        self.show_frame_axes = visible

        if not visible:
            # Remove all frame axes from view
            self.clear_frame_axes()
        # If visible, they will be redrawn on next update_time()

    def clear_frame_axes(self) -> None:
        """Remove all drawn user-defined frame axes."""
        for frame_name in list(self.frame_axes.keys()):
            for axis_line in self.frame_axes[frame_name].values():
                try:
                    self.view.removeItem(axis_line)
                except Exception:
                    pass
        self.frame_axes.clear()

        # Also remove frame labels
        for label in self.frame_labels.values():
            try:
                self.view.removeItem(label)
            except Exception:
                pass
        self.frame_labels.clear()

    def _update_frame_axes(self, pos_dict: Dict[str, np.ndarray]) -> None:
        """Update frame axes visualization for all defined frames.

        Draws coordinate frame axes at body part positions when frame names
        match body part names, or at origin for frames without matching parts.

        Args:
            pos_dict: Dictionary mapping part names to their 3D positions.
        """
        if not self.show_frame_axes or not self.frames:
            return

        # Track which frames we've drawn
        drawn_frames = set()

        for frame_name, frame_info in self.frames.items():
            # Skip the implicit lab/world frame
            if frame_name == "lab":
                continue

            # Find position for this frame (match to body part if possible)
            if frame_name in pos_dict:
                pos = pos_dict[frame_name]
            else:
                # Try to find a matching part name (case-insensitive)
                pos = None
                for part_name, part_pos in pos_dict.items():
                    if part_name.lower() == frame_name.lower():
                        pos = part_pos
                        break

                # No matching part - draw at origin with small z offset
                if pos is None:
                    idx = list(self.frames.keys()).index(frame_name)
                    pos = np.array([0.0, 0.0, 0.1 * idx], dtype=float)

            total_offset = self._frame_offset(frame_name)
            self.draw_frame_at_position(frame_name, pos, total_offset)
            drawn_frames.add(frame_name)

        # Remove frame axes for frames that no longer exist
        for frame_name in list(self.frame_axes.keys()):
            if frame_name not in drawn_frames:
                for axis_line in self.frame_axes[frame_name].values():
                    try:
                        self.view.removeItem(axis_line)
                    except Exception:
                        pass
                del self.frame_axes[frame_name]

                if frame_name in self.frame_labels:
                    try:
                        self.view.removeItem(self.frame_labels[frame_name])
                    except Exception:
                        pass
                    del self.frame_labels[frame_name]

    def _heading_from_rotation(self, part: str, rot: np.ndarray, row: pd.Series) -> Optional[float]:
        """Derive heading from rotation matrix or fallbacks.

        Issue 3 fix: Returns None when heading extraction fails instead of 0.0,
        allowing callers to distinguish 'no data' from 'heading is 0 degrees'.
        """
        try:
            fwd = rot[2]  # forward row
            return math.degrees(math.atan2(fwd[1], fwd[0]))
        except Exception:
            pass
        # Try to get heading from column, return None if not available
        heading_col = f"{part}_heading_deg"
        if heading_col in row:
            val = row[heading_col]
            # Check for NaN values which indicate missing data
            if pd.notna(val):
                return float(val)
        return None

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

    def _remove_arrow(self, part: str) -> None:
        """Remove arrow for a part when heading is unavailable (Issue 3 fix)."""
        if part in self.arrows:
            try:
                self.view.removeItem(self.arrows[part])
            except Exception:
                pass
            del self.arrows[part]

    def _frame_offset(self, part: str) -> float:
        """Compute total heading offset by walking the parent chain.

        This method traverses the frame hierarchy to accumulate offsets from
        the given frame through all its ancestors up to the root (lab frame).

        Args:
            part: The frame/body part name to compute offset for.

        Returns:
            Total accumulated offset in degrees from all frames in the chain.
        """
        return self._get_total_offset(part, set())

    def _get_total_offset(self, frame_name: str, visited: set) -> float:
        """Recursive helper to compute total offset with cycle detection.

        Args:
            frame_name: The frame to compute offset for.
            visited: Set of already-visited frame names to detect cycles.

        Returns:
            Accumulated offset in degrees. Returns 0.0 if frame not found
            or if a cycle is detected.
        """
        # Cycle detection: if we've already visited this frame, stop recursion
        if frame_name in visited:
            return 0.0

        # Frame not in our registry: return 0
        if frame_name not in self.frames:
            return 0.0

        visited.add(frame_name)

        info = self.frames.get(frame_name, {})
        offset = float(info.get("offset", 0.0))
        parent = info.get("parent", "")

        # If there's a valid parent frame, recursively add its offset
        if parent and parent in self.frames:
            offset += self._get_total_offset(parent, visited)

        return offset

    def get_frame_chain(self, part: str) -> List[str]:
        """Return list of frames from root to this part.

        Useful for debugging and visualization of the kinematic chain.
        Includes cycle detection to prevent infinite loops.

        Args:
            part: The frame/body part name to trace back to root.

        Returns:
            List of frame names from root (first) to the given part (last).
            Empty list if part is not in frames registry.
        """
        chain = []
        current = part
        visited = set()
        while current and current not in visited:
            visited.add(current)
            if current in self.frames:
                chain.append(current)
                current = self.frames[current].get("parent", "")
            else:
                break
        return list(reversed(chain))

    def detect_frame_cycle(self, frame_name: str) -> bool:
        """Check if a frame is part of a cycle in the parent chain.

        Args:
            frame_name: The frame to check for cycles.

        Returns:
            True if a cycle is detected, False otherwise.
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
        """Draw motion trails for each part.

        Issue 1 fix: Prunes trail_history entries for parts no longer in current
        positions dict to prevent memory leak over long sessions when user changes
        which parts are displayed.
        """
        # Issue 1 fix: Prune stale trail entries for parts no longer displayed
        # First, identify parts that are in trail_history but not in current positions
        stale_parts = [part for part in self.trail_history if part not in positions]
        for part in stale_parts:
            # Remove the trail graphics item from the view first
            if part in self.trails:
                try:
                    self.view.removeItem(self.trails[part])
                except Exception:
                    pass
                del self.trails[part]
            # Then delete from trail_history dict
            del self.trail_history[part]

        # Update trails for current parts
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

    # ------------------------------------------------------------------
    # Camera Controls
    # ------------------------------------------------------------------
    def set_view_preset(self, preset: str) -> None:
        """Set camera to predefined viewpoint.

        Args:
            preset: One of 'top', 'front', 'side', 'left', 'back', 'isometric', 'reset'

        pyqtgraph.opengl uses elevation (vertical angle) and azimuth (horizontal angle):
        - elevation: 0 = horizontal, 90 = looking straight down
        - azimuth: 0 = looking along +X, 90 = looking along +Y
        """
        presets = {
            "top": {"elevation": 90, "azimuth": 0, "distance": 6},
            "front": {"elevation": 0, "azimuth": 0, "distance": 6},
            "back": {"elevation": 0, "azimuth": 180, "distance": 6},
            "side": {"elevation": 0, "azimuth": 90, "distance": 6},
            "left": {"elevation": 0, "azimuth": -90, "distance": 6},
            "isometric": {"elevation": 30, "azimuth": 45, "distance": 6},
            "reset": {"elevation": 30, "azimuth": 45, "distance": 6, "center": (0, 0, 0)},
        }
        if preset in presets:
            params = presets[preset]
            self.view.opts["elevation"] = params["elevation"]
            self.view.opts["azimuth"] = params["azimuth"]
            self.view.opts["distance"] = params["distance"]
            if "center" in params:
                self.view.opts["center"] = QtGui.QVector3D(*params["center"])
            self.view.update()

    def zoom_in(self, factor: float = 0.8) -> None:
        """Zoom in by reducing camera distance.

        Args:
            factor: Multiplier for distance (< 1.0 zooms in). Default 0.8.
        """
        self.view.opts["distance"] *= factor
        self.view.update()

    def zoom_out(self, factor: float = 1.25) -> None:
        """Zoom out by increasing camera distance.

        Args:
            factor: Multiplier for distance (> 1.0 zooms out). Default 1.25.
        """
        self.view.opts["distance"] *= factor
        self.view.update()

    def get_view_state(self) -> dict:
        """Get current camera state for save/restore.

        Returns:
            Dictionary with elevation, azimuth, and distance values.
        """
        return {
            "elevation": self.view.opts["elevation"],
            "azimuth": self.view.opts["azimuth"],
            "distance": self.view.opts["distance"],
        }

    def set_view_state(self, state: dict) -> None:
        """Restore camera state from saved dict.

        Args:
            state: Dictionary with elevation, azimuth, and/or distance keys.
        """
        for key in ["elevation", "azimuth", "distance"]:
            if key in state:
                self.view.opts[key] = state[key]
        self.view.update()

