"""Scenario serialization helpers.

A World fully describes a scenario, so saving and loading a scenario is just
serializing the World dictionary. JSON is always available; YAML is used when
PyYAML is installed for a more readable file.
"""

from __future__ import annotations

import json
import os

from .world import World


def save_scenario(world: World, path: str) -> str:
    data = world.to_dict()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, sort_keys=False)
            return path
        except ImportError:
            path = os.path.splitext(path)[0] + ".json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


def load_scenario(path: str) -> World:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    return World.from_dict(data)
