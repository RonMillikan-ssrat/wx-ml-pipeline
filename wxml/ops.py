"""Task logic for the Airflow DAGs, kept here so it can be tested without Airflow.

Every function takes the date it works on as an argument and never reads the
wall clock to decide what to process. That is what makes backfills work: a run
for 2026-10-03 does exactly what it would have done on 2026-10-03.
"""
from datetime import date, timedelta

import joblib
import pandas as pd

from . import config, data, model, monitor, registry, store
from .features import FEATURES, build_features

WINDOW_DAYS = 14     # history needed to build one day's features (lags, interpolation)


# ---------------------------------------------------------------- daily
def validate_day(d: date) -> dict:
    df = data.load_range(d - timedelta(days=WINDOW_DAYS - 1), d)
    return data.validate_window(df, d, WINDOW_DAYS)


def verify_day(d: date) -> dict:
    """Score the forecast that was issued for day d (by the previous day's run)."""
    fc = store.read_json(store.forecast_path(d))
    if fc is None:
        return {"date": d.isoformat(), "status": "no_forecast"}
    obs = data.load_range(d, d)
    if obs.empty or pd.isna(obs.tmax.iloc[0]):
        raise data.DataValidationError(f"no observed tmax for {d}")
    observed = float(obs.tmax.iloc[0])
    rec = {
        "date": d.isoformat(), "status": "verified", "observed_F": observed,
        "forecast_F": fc["tmax_forecast_F"], "persistence_F": fc["persistence_F"],
        "error_model_F": fc["tmax_forecast_F"] - observed,
        "error_persistence_F": fc["persistence_F"] - observed,
        "model_version": fc["model_version"], "in_sample": fc["in_sample"],
    }
    store.write_json(store.verification_path(d), rec)
    return rec


def check_skill(d: date) -> dict:
    version = registry.alias_version(config.PROD_ALIAS)
    if version is None:
        return {"n": 0, "model_version": None, "degraded": False}
    stats = monitor.rolling_skill(d, version)
    stats["degraded"] = monitor.is_degraded(stats)
    return stats


def forecast_day(d: date, run_id: str) -> dict:
    """Using observations through day d, forecast Tmax for d+1 with the production model."""
    m, version, tags = registry.load(config.PROD_ALIAS)
    raw = data.load_range(d - timedelta(days=WINDOW_DAYS - 1), d)
    feats = build_features(raw, with_target=False)
    if feats.empty or feats.index[-1].date() != d:
        raise data.DataValidationError(f"could not build complete features for {d}")
    row = feats.iloc[[-1]][FEATURES]
    valid = d + timedelta(days=1)
    train_end = date.fromisoformat(tags["train_end"])
    rec = {
        "valid_date": valid.isoformat(), "issued_from": d.isoformat(),
        "tmax_forecast_F": float(m.predict(row)[0]),
        "persistence_F": float(row.tmax_l0.iloc[0]),
        "model_version": str(version), "model_train_end": train_end.isoformat(),
        "in_sample": valid < train_end,            # backfill into the training period
        "airflow_run_id": run_id,
        "features": {k: float(v) for k, v in row.iloc[0].items()},
    }
    store.write_json(store.forecast_path(valid), rec)
    return rec


# ---------------------------------------------------------------- weekly retrain
def candidate_dir(run_id: str):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
    return config.DATA_DIR / "candidates" / safe


def train_candidate(end: date, run_id: str) -> dict:
    """Train on everything before the holdout; score candidate AND production on the holdout.

    holdout = the last HOLDOUT_DAYS days up to `end`. Both models are scored on
    the same days, so the comparison is fair (champion/challenger).
    """
    holdout_start = end - timedelta(days=config.HOLDOUT_DAYS)
    raw = data.load_range(date(config.TRAIN_START_YEAR, 1, 1), end)
    f = build_features(raw)
    tr = f[f.index < pd.Timestamp(holdout_start)]
    te = f[f.index >= pd.Timestamp(holdout_start)]
    if len(te) < 30:
        raise data.DataValidationError(f"holdout has only {len(te)} rows")
    m = model.train(tr)
    cand = model.evaluate(m, te)

    prod_version, prod = registry.alias_version(config.PROD_ALIAS), None
    if prod_version is not None:
        prod_model, _, _ = registry.load(config.PROD_ALIAS)
        prod = model.evaluate(prod_model, te)

    out = candidate_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, out / "model.joblib")              # same path on retry -> overwrite
    result = {"candidate": cand, "production": prod, "production_version": prod_version,
              "train_rows": int(len(tr)), "train_end": holdout_start.isoformat(),
              "holdout_end": end.isoformat(), "model_path": str(out / "model.joblib")}
    store.write_json(out / "metrics.json", result)
    return result


def register_candidate(result: dict, run_id: str) -> str:
    """Register the candidate as a new model version. A retry reuses its earlier version."""
    existing = registry.find_version_by_tag("airflow_run_id", run_id)
    if existing:
        return existing
    m = joblib.load(result["model_path"])
    c = result["candidate"]
    return registry.register(
        m,
        metrics={"skill": c["skill"], "rmse_F": c["rmse_F"],
                 "rmse_persistence_F": c["rmse_persistence_F"], "n_test": c["n_test"]},
        params={"max_iter": config.MAX_ITER, "learning_rate": config.LEARNING_RATE,
                "train_start": f"{config.TRAIN_START_YEAR}-01-01", "train_end": result["train_end"],
                "holdout_end": result["holdout_end"], "source": config.SOURCE,
                "station": config.STATION_ID},
        tags={"airflow_run_id": run_id, "train_end": result["train_end"],
              "holdout_end": result["holdout_end"], "skill": round(c["skill"], 4),
              "status": "candidate", "source": config.SOURCE},
    )


def gate(result: dict) -> tuple[bool, str]:
    """Promote only if the candidate beats persistence by SKILL_GATE and is no worse than production."""
    c, p = result["candidate"], result["production"]
    if c["skill"] <= config.SKILL_GATE:
        return False, f"skill {c['skill']:.3f} <= gate {config.SKILL_GATE}"
    if p is not None and c["rmse_F"] > p["rmse_F"]:
        return False, (f"candidate RMSE {c['rmse_F']:.3f}F worse than production "
                       f"v{result['production_version']} {p['rmse_F']:.3f}F on the same holdout")
    vs = "no production model yet" if p is None else \
        f"RMSE {c['rmse_F']:.3f}F vs production {p['rmse_F']:.3f}F"
    return True, f"skill {c['skill']:.3f} > {config.SKILL_GATE}; {vs}"
