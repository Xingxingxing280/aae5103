from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with (ROOT / "config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def data_root() -> Path:
    return (ROOT / load_config()["data_root"]).resolve()


def models_dir() -> Path:
    path = ROOT / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reports_dir() -> Path:
    path = ROOT / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    path = reports_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def predictions_dir() -> Path:
    path = ROOT / "data" / "predictions"
    path.mkdir(parents=True, exist_ok=True)
    return path
