"""Simple plugin loader for derived metrics and filters."""
from __future__ import annotations

import json
import os


class PluginManager:
    def __init__(self, directory: str = "plugins") -> None:
        self.directory = directory
        self.plugins: list[dict] = []
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
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                self.plugins.append(data)
            except Exception:
                continue

    def menu_entries(self) -> list[str]:
        names = []
        for p in self.plugins:
            name = p.get("name") or p.get("id")
            if name:
                names.append(name)
        return names

    def get_plugin(self, name: str) -> dict:
        for p in self.plugins:
            if p.get("name") == name or p.get("id") == name:
                return p
        return {}

