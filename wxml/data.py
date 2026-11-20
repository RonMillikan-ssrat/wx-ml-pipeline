"""GSOD observations: fetch, cache, check availability, validate.

Real data comes from NCEI's per-station, per-year CSV files
(https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/<year>/<station>.csv),
the same GSOD product the KFP pipeline reads from BigQuery, with the same units
(degrees F, mb, knots, inches). No GCP credentials are needed.

Synthetic data is a deterministic offline stand-in for development and drills.
"""
import io
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config, store

COLUMNS = ["date", "temp", "tmax", "tmin", "dewp", "slp", "wdsp", "prcp"]
NCEI_COLUMNS = {"DATE": "date", "TEMP": "temp", "MAX": "tmax", "MIN": "tmin",
                "DEWP": "dewp", "SLP": "slp", "WDSP": "wdsp", "PRCP": "prcp"}
SENTINELS = {9999.9: np.nan, 999.9: np.nan, 99.99: np.nan}   # GSOD "no report" values


class DataValidationError(ValueError):
    """Bad data. Retrying will not fix it, so tasks treat this as a hard failure."""


# ---------------------------------------------------------------- synthetic
_SYNTH = None


def _synthetic_series() -> pd.DataFrame:
    """One fixed series for 2005-2035 (seed 0), so any slice is reproducible.

    A pressure anomaly p follows an AR(1) process; falling pressure warms the
    next day. Lag features plus pressure tendency therefore carry real signal,
    and the model beats persistence (skill around 0.3), roughly like IAH.
    """
    global _SYNTH
    if _SYNTH is None:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2005-01-01", "2035-12-31", freq="D")
        n = len(idx)
        doy = idx.dayofyear.to_numpy()
        seas = 80 + 15 * np.sin(2 * np.pi * (doy - 110) / 365.25)
        p, a = np.zeros(n), np.zeros(n)
        for i in range(2, n):
            p[i] = 0.85 * p[i - 1] + rng.normal(0, 3)
            a[i] = 0.6 * a[i - 1] - 0.9 * (p[i - 1] - p[i - 2]) + rng.normal(0, 2.5)
        tmax = seas + a
        _SYNTH = pd.DataFrame({
            "date": idx.date, "tmax": tmax,
            "tmin": tmax - 18 + rng.normal(0, 2, n), "temp": tmax - 9,
            "dewp": tmax - 20 + rng.normal(0, 3, n),
            "slp": 1015 + p + rng.normal(0, 0.5, n),
            "wdsp": rng.gamma(2, 3, n), "prcp": rng.exponential(0.1, n),
        })[COLUMNS]
    return _SYNTH


def _synthetic_year(year: int, now: datetime) -> pd.DataFrame:
    df = _synthetic_series()
    last_published = (now - timedelta(days=config.SYNTH_LAG_DAYS)).date()
    keep = (pd.to_datetime(df.date).dt.year == year) & (df.date <= last_published)
    out = df[keep].reset_index(drop=True)
    if config.DRILL_BAD_OBS_DATE:                     # failure drill, see config.py
        out.loc[out.date == date.fromisoformat(config.DRILL_BAD_OBS_DATE), "tmax"] = 150.0
    return out


# ---------------------------------------------------------------- NCEI
def _ncei_year(year: int) -> pd.DataFrame:
    import requests   # imported here so offline/synthetic use needs no network stack
    url = config.NCEI_URL.format(year=year, station=config.STATION_ID)
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        return pd.DataFrame(columns=COLUMNS)          # year not published yet
    r.raise_for_status()                              # 5xx etc. -> exception -> Airflow retries
    raw = pd.read_csv(io.StringIO(r.text), dtype=str, skipinitialspace=True)
    missing = set(NCEI_COLUMNS) - set(raw.columns)
    if missing:
        raise DataValidationError(f"NCEI file {url} is missing columns {sorted(missing)}")
    df = raw[list(NCEI_COLUMNS)].rename(columns=NCEI_COLUMNS)
    df["date"] = pd.to_datetime(df["date"].str.strip()).dt.date
    for c in COLUMNS[1:]:
        df[c] = pd.to_numeric(df[c].str.strip(), errors="coerce")
    return df.replace(SENTINELS)[COLUMNS]


# ---------------------------------------------------------------- cache
def _cache_path(year: int):
    return config.DATA_DIR / "raw" / f"gsod_{config.STATION_ID}_{year}_{config.SOURCE}.csv"


def fetch_year(year: int, force: bool = False, now: datetime | None = None) -> pd.DataFrame:
    """Return one station-year, using a local cache.

    Past years are downloaded once. The current (or future) year is
    re-downloaded when the cache is older than REFRESH_HOURS or force=True,
    because NCEI keeps appending to it.
    """
    now = now or datetime.now(timezone.utc)
    path = _cache_path(year)
    still_growing = year >= now.year - (1 if now.month == 1 else 0)
    stale = (not path.exists()
             or (still_growing and time.time() - path.stat().st_mtime > config.REFRESH_HOURS * 3600))
    if force or stale:
        df = _synthetic_year(year, now) if config.SOURCE == "synthetic" else _ncei_year(year)
        store.atomic_write_text(path, df.to_csv(index=False))
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_range(start: date, end: date, force_refresh: bool = False) -> pd.DataFrame:
    """Daily rows with start <= date <= end, from the yearly caches."""
    frames = [fetch_year(y, force=force_refresh) for y in range(start.year, end.year + 1)]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.date >= start) & (df.date <= end)]
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def observation_available(d: date) -> bool:
    """True once day d has a published, non-missing Tmax. Forces a fresh download."""
    df = fetch_year(d.year, force=True)
    row = df[df.date == d]
    return len(row) == 1 and pd.notna(row.tmax.iloc[0])


# ---------------------------------------------------------------- validation
def validate_window(df: pd.DataFrame, d: date, window_days: int) -> dict:
    """Checks before the data feeds a forecast. Raises DataValidationError."""
    problems = []
    today = df[df.date == d]
    if len(today) != 1:
        problems.append(f"no row for {d}")
    elif today[["tmax", "tmin", "slp"]].isna().any(axis=None):
        problems.append(f"{d} is missing tmax/tmin/slp: {today.iloc[0].to_dict()}")
    bad_range = df[(df.tmax < -40) | (df.tmax > 130) | (df.tmin < -60) | (df.tmin > 110)]
    if len(bad_range):
        problems.append(f"out-of-range temperatures on {list(bad_range.date)}")
    inverted = df[df.tmin > df.tmax + 0.05]
    if len(inverted):
        problems.append(f"tmin > tmax on {list(inverted.date)}")
    bad_slp = df[(df.slp < 900) | (df.slp > 1090)]
    if len(bad_slp):
        problems.append(f"sea-level pressure out of range on {list(bad_slp.date)}")
    days_present = df.date.nunique()
    if days_present < window_days * 0.7:
        problems.append(f"only {days_present}/{window_days} days present in the window")
    if problems:
        raise DataValidationError("; ".join(problems))
    return {"date": d.isoformat(), "rows": int(len(df)), "days_present": int(days_present),
            "tmax": float(today.tmax.iloc[0])}
