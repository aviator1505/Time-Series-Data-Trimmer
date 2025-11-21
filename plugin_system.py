"""Simple plugin loader for derived metrics and filters."""
from __future__ import annotations

import json
import os
from typing import Dict, List


class PluginManager:
    def __init__(self, directory: str = "plugins") -> None:
        self.directory = directory
        self.plugins: List[Dict] = []
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory, exist_ok=True)

    def load_plugins(self) -> None:
        self.plugins.clear()
        if not os.path.isdir(self.directory):
            return
        for fname in os.listdir(self.directory):
            if not fname.lower().endswith((".json", ".plugin")):
                continue
            path = os.path.join(self.directory, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.plugins.append(data)
            except Exception:
                continue

    def menu_entries(self) -> List[str]:
        names = []
        for p in self.plugins:
            name = p.get("name") or p.get("id")
            if name:
                names.append(name)
        return names

    def get_plugin(self, name: str) -> Dict:
        for p in self.plugins:
            if p.get("name") == name or p.get("id") == name:
                return p
        return {}

