"""ProjectManager groups multiple trials and recipes."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


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
    operations: List[Dict]


class ProjectManager:
    def __init__(self) -> None:
        self.project_path: Optional[str] = None
        self.trials: List[TrialEntry] = []
        self.recipes: List[Recipe] = []
        self.preferences: Dict = {
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

    def add_trials_bulk(self, entries: List[TrialEntry]) -> None:
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
            "trials": [asdict(t) for t in self.trials],
            "recipes": [asdict(r) for r in self.recipes],
            "preferences": self.preferences,
        }
        with open(self.project_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.project_path = path
        self.trials = [TrialEntry(**t) for t in data.get("trials", [])]
        self.recipes = [Recipe(**r) for r in data.get("recipes", [])]
        self.preferences = data.get("preferences", self.preferences)

    def export_summary(self) -> List[Dict]:
        return [asdict(t) for t in self.trials]

    def add_recipe(self, recipe: Recipe) -> None:
        self.recipes.append(recipe)

