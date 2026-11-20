"""Idempotent, atomic storage for forecasts and verifications.

Each record is one JSON file keyed by date, for example
    data/forecasts/valid_date=2026-11-21.json
Re-running a task for the same date overwrites the same file, so retries and
backfills never create duplicates. Writes go to a temporary file first and are
renamed into place with os.replace, which is atomic on POSIX: a task killed
mid-write leaves the old file intact rather than a half-written one.
"""
import json
import os
import tempfile
from datetime import date
from pathlib import Path

from . import config


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_json(path: Path, obj: dict) -> Path:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    return path


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def forecast_path(valid_date: date) -> Path:
    return config.DATA_DIR / "forecasts" / f"valid_date={valid_date.isoformat()}.json"


def verification_path(d: date) -> Path:
    return config.DATA_DIR / "verification" / f"date={d.isoformat()}.json"


def list_verifications(start: date, end: date) -> list[dict]:
    """All verification records with start <= date <= end."""
    out = []
    folder = config.DATA_DIR / "verification"
    if not folder.exists():
        return out
    for p in sorted(folder.glob("date=*.json")):
        d = date.fromisoformat(p.stem.split("=", 1)[1])
        if start <= d <= end:
            out.append(json.loads(p.read_text()))
    return out
