"""Ongoing skill monitoring from verification records.

A forecast can only be scored once the day it predicted has been observed, so
skill is measured after the fact, one day at a time. The rolling skill of the
production model over the last N verified days is the production equivalent of
the evaluate step in the training pipeline.

Only out-of-sample days count: a forecast for a date inside the model's own
training period (possible in backfills) would flatter the model.
"""
from datetime import date, timedelta

import numpy as np

from . import config, store


def rolling_skill(end: date, model_version: str, window_days: int | None = None) -> dict:
    window_days = window_days or config.ROLLBACK_WINDOW_DAYS
    start = end - timedelta(days=window_days - 1)
    recs = [r for r in store.list_verifications(start, end)
            if r.get("model_version") == str(model_version) and not r.get("in_sample")]
    n = len(recs)
    if n == 0:
        return {"n": 0, "window_start": start.isoformat(), "window_end": end.isoformat(),
                "model_version": str(model_version)}
    e_m = np.array([r["error_model_F"] for r in recs])
    e_p = np.array([r["error_persistence_F"] for r in recs])
    rmse, rmse_p = float(np.sqrt(np.mean(e_m ** 2))), float(np.sqrt(np.mean(e_p ** 2)))
    skill = 1 - rmse / rmse_p if rmse_p > 0 else float("nan")
    return {"n": n, "rmse_F": rmse, "rmse_persistence_F": rmse_p, "skill": skill,
            "window_start": start.isoformat(), "window_end": end.isoformat(),
            "model_version": str(model_version)}


def is_degraded(stats: dict) -> bool:
    """Degraded = enough evidence AND rolling skill below the rollback threshold."""
    return stats["n"] >= config.ROLLBACK_MIN_N and stats["skill"] < config.ROLLBACK_SKILL
