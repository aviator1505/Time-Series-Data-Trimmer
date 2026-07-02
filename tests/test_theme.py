"""Tests for Qt-native theming (theme.py)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from theme import THEME_CHOICES, apply_theme, effective_scheme


def test_apply_dark_theme(qapp):
    assert apply_theme(qapp, "Dark") == "dark"
    assert effective_scheme(qapp) == "dark"


def test_apply_light_theme(qapp):
    assert apply_theme(qapp, "Light") == "light"
    assert effective_scheme(qapp) == "light"


def test_apply_system_theme_returns_valid_scheme(qapp):
    assert apply_theme(qapp, "System") in ("dark", "light")
    # restore light for other tests
    apply_theme(qapp, "Light")


def test_theme_input_is_case_insensitive(qapp):
    assert apply_theme(qapp, "dArK") == "dark"
    apply_theme(qapp, "Light")


def test_none_theme_falls_back_to_system(qapp):
    assert apply_theme(qapp, None) in ("dark", "light")
    apply_theme(qapp, "Light")


def test_theme_choices_exposed():
    assert THEME_CHOICES == ("System", "Light", "Dark")
