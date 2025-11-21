"""ProjectManager groups multiple trials and recipes."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class TrialEntry:
    path: str
    participant: str = ""
    condition: str = ""
    status: str = "unloaded"  # unloaded / loaded / cleaned / exported
    notes: str = ""


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

