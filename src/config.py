"""Read configs/config.yaml and resolve the paths in it."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(Path(path) if path else DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Relative paths in the config are relative to the repository root, so every command
    behaves the same whichever folder it is started from."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def reports_dir(cfg: dict) -> Path:
    return resolve(cfg["paths"]["reports"])
