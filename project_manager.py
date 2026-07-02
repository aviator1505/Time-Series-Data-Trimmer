"""ProjectManager groups multiple trials and recipes."""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, fields
from typing import Any

# Schema version for project files - increment when format changes
PROJECT_SCHEMA_VERSION = 1


@dataclass
class ParsedFilename:
    """Result of parsing a trial filename."""
    trial_number: int
    session: int
    participant: str
    condition: str
    angle: int
    parse_success: bool
    original_filename: str


def parse_trial_filename(filepath: str) -> ParsedFilename:
    """
    Parse trial metadata from filename pattern: trial_session_participant_condition_angle.csv

    Example: 8_1_P13_Stand_45.csv -> (8, 1, "P13", "Stand", 45)

    Args:
        filepath: Full path to CSV file

    Returns:
        ParsedFilename with extracted fields and parse_success flag
    """
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]

    # Pattern: trial_session_participant_condition_angle
    pattern = r'^(\d+)_(\d+)_(P\d+)_(Sit|Stand|Swivel)_(\d+)$'
    match = re.match(pattern, name_without_ext, re.IGNORECASE)

    if match:
        return ParsedFilename(
            trial_number=int(match.group(1)),
            session=int(match.group(2)),
            participant=match.group(3).upper(),
            condition=match.group(4).capitalize(),
            angle=int(match.group(5)),
            parse_success=True,
            original_filename=filename
        )
    else:
        return ParsedFilename(
            trial_number=0,
            session=0,
            participant="",
            condition="",
            angle=0,
            parse_success=False,
            original_filename=filename
        )


@dataclass
class TrialEntry:
    path: str
    participant: str = ""
    condition: str = ""
    status: str = "unloaded"  # unloaded / loaded / cleaned / exported
    summary: str = ""
    notes: str = ""
    # Additional parsed metadata fields
    trial_number: int = 0
    session: int = 0
    angle: int = 0


@dataclass
class Recipe:
    name: str
    operations: list[dict]


class ProjectManager:
    def __init__(self) -> None:
        self.project_path: str | None = None
        self.trials: list[TrialEntry] = []
        self.recipes: list[Recipe] = []
        self.preferences: dict = {
            "default_fs": 120.0,
            "default_output_dir": os.getcwd(),
        }

    def new_project(self, path: str) -> None:
        self.project_path = path
        self.trials = []
        self.recipes = []
        self.save()

    def add_trial(self, path: str, participant: str = "", condition: str = "") -> None:
        self.trials.append(TrialEntry(path=path, participant=participant, condition=condition))

    def add_trials_bulk(self, entries: list[TrialEntry]) -> None:
        """Add multiple trials at once."""
        self.trials.extend(entries)

    def update_status(self, path: str, status: str, summary: str = "") -> None:
        for t in self.trials:
            if t.path == path:
                t.status = status
                if summary:
                    t.summary = summary
                break

    def save(self) -> None:
        if not self.project_path:
            return
        data = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "trials": [asdict(t) for t in self.trials],
            "recipes": [asdict(r) for r in self.recipes],
            "preferences": self.preferences,
        }

        # Atomic write: write to temp file, then rename
        project_dir = os.path.dirname(self.project_path) or "."
        fd, temp_path = tempfile.mkstemp(suffix=".json", dir=project_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            # Atomic rename (works on same filesystem)
            shutil.move(temp_path, self.project_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _migrate_schema(self, data: dict, from_version: int) -> dict:
        """Migrate data from older schema versions.

        Args:
            data: Project data dictionary
            from_version: Source schema version

        Returns:
            Migrated data dictionary
        """
        # Currently at version 1, no migrations needed yet
        # Future migrations would be added here:
        # if from_version < 2:
        #     # Migrate from v1 to v2
        #     data = self._migrate_v1_to_v2(data)
        return data

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.project_path = path

        # Check schema version and migrate if needed
        schema_version = data.get("schema_version", 0)
        if schema_version < PROJECT_SCHEMA_VERSION:
            data = self._migrate_schema(data, schema_version)

        # Safely load trials with unknown field tolerance
        trials_data = data.get("trials", [])
        self.trials = []
        known_trial_fields = {f.name for f in fields(TrialEntry)}
        for t in trials_data:
            try:
                filtered = {k: v for k, v in t.items() if k in known_trial_fields}
                self.trials.append(TrialEntry(**filtered))
            except (TypeError, ValueError):
                # Skip malformed trial entries
                continue

        # Same for recipes
        recipes_data = data.get("recipes", [])
        self.recipes = []
        known_recipe_fields = {f.name for f in fields(Recipe)}
        for r in recipes_data:
            try:
                filtered = {k: v for k, v in r.items() if k in known_recipe_fields}
                self.recipes.append(Recipe(**filtered))
            except (TypeError, ValueError):
                # Skip malformed recipe entries
                continue

        # Merge preferences to preserve new defaults
        self.preferences = {**self.preferences, **data.get("preferences", {})}

    def export_summary(self) -> list[dict]:
        return [asdict(t) for t in self.trials]

    def add_recipe(self, recipe: Recipe) -> None:
        self.recipes.append(recipe)


# ---------------------------------------------------------------------------
# Global signal preset persistence (independent of projects)
# ---------------------------------------------------------------------------


def _get_presets_path() -> pathlib.Path:
    """Get platform-appropriate path for signal presets."""
    if os.name == 'nt':  # Windows
        base = pathlib.Path(os.environ.get('APPDATA', pathlib.Path.home()))
    else:  # macOS/Linux
        base = pathlib.Path.home() / '.config'

    app_dir = base / 'TimeSeriesDataTrimmer'
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir / 'signal_presets.json'


SIGNAL_PRESETS_FILE = _get_presets_path()


def load_signal_presets() -> dict[str, dict]:
    """Load signal presets from disk."""
    if not SIGNAL_PRESETS_FILE.is_file():
        return {}
    try:
        with open(SIGNAL_PRESETS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_signal_presets(presets: dict[str, dict]) -> None:
    """Save signal presets to disk with atomic write."""
    try:
        preset_dir = SIGNAL_PRESETS_FILE.parent
        preset_dir.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        fd, temp_path = tempfile.mkstemp(suffix=".json", dir=str(preset_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(presets, f, indent=2)
            shutil.move(temp_path, str(SIGNAL_PRESETS_FILE))
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        print(f"Failed to save presets: {e}")


def load_ui_state() -> dict[str, Any]:
    """Load UI state (splitter positions) from signal_presets.json."""
    presets = load_signal_presets()
    return presets.get("__ui_state__", {})


def save_ui_state(state: dict[str, Any]) -> None:
    """Save UI state to signal_presets.json under __ui_state__ key."""
    presets = load_signal_presets()
    presets["__ui_state__"] = state
    save_signal_presets(presets)

