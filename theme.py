"""Application theming: Fusion style with light/dark/system color schemes.

Uses Qt's built-in color-scheme support (Qt >= 6.8) rather than a
third-party stylesheet package, so theming adds no dependency. On older
Qt a hand-built dark palette is used as fallback.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

THEME_CHOICES = ("System", "Light", "Dark")


def apply_theme(app: QtWidgets.QApplication, theme: str) -> str:
    """Apply the requested theme ("System"/"Light"/"Dark", case-insensitive).

    Returns the effective scheme, "dark" or "light", so callers can restyle
    non-Qt surfaces (e.g. pyqtgraph plot backgrounds) to match.
    """
    normalized = (theme or "system").strip().lower()
    app.setStyle("Fusion")
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):
        scheme = {
            "light": QtCore.Qt.ColorScheme.Light,
            "dark": QtCore.Qt.ColorScheme.Dark,
        }.get(normalized, QtCore.Qt.ColorScheme.Unknown)  # Unknown follows the OS
        hints.setColorScheme(scheme)
    # Some platforms (offscreen, older desktops) ignore setColorScheme; fall
    # back to an explicit palette whenever the request didn't take effect.
    if normalized == "dark" and effective_scheme(app) != "dark":
        app.setPalette(_dark_palette())
    elif normalized == "light" and effective_scheme(app) != "light":
        app.setPalette(app.style().standardPalette())
    return effective_scheme(app)


def effective_scheme(app: QtWidgets.QApplication) -> str:
    """Return "dark" or "light" for the palette currently in effect."""
    hints = app.styleHints()
    if hasattr(hints, "colorScheme"):
        if hints.colorScheme() == QtCore.Qt.ColorScheme.Dark:
            return "dark"
        if hints.colorScheme() == QtCore.Qt.ColorScheme.Light:
            return "light"
    window = app.palette().color(QtGui.QPalette.ColorRole.Window)
    return "dark" if window.lightness() < 128 else "light"


def _dark_palette() -> QtGui.QPalette:
    """Manual dark palette for Qt versions without setColorScheme."""
    p = QtGui.QPalette()
    window = QtGui.QColor(43, 43, 43)
    base = QtGui.QColor(30, 30, 30)
    text = QtGui.QColor(221, 221, 221)
    highlight = QtGui.QColor(42, 130, 218)
    p.setColor(QtGui.QPalette.ColorRole.Window, window)
    p.setColor(QtGui.QPalette.ColorRole.WindowText, text)
    p.setColor(QtGui.QPalette.ColorRole.Base, base)
    p.setColor(QtGui.QPalette.ColorRole.AlternateBase, window)
    p.setColor(QtGui.QPalette.ColorRole.ToolTipBase, base)
    p.setColor(QtGui.QPalette.ColorRole.ToolTipText, text)
    p.setColor(QtGui.QPalette.ColorRole.Text, text)
    p.setColor(QtGui.QPalette.ColorRole.Button, window)
    p.setColor(QtGui.QPalette.ColorRole.ButtonText, text)
    p.setColor(QtGui.QPalette.ColorRole.Link, highlight)
    p.setColor(QtGui.QPalette.ColorRole.Highlight, highlight)
    p.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))
    disabled = QtGui.QColor(128, 128, 128)
    for role in (QtGui.QPalette.ColorRole.WindowText, QtGui.QPalette.ColorRole.Text,
                 QtGui.QPalette.ColorRole.ButtonText):
        p.setColor(QtGui.QPalette.ColorGroup.Disabled, role, disabled)
    return p
