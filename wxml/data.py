"""Daily observations: fetch, summarize, cache, check availability, validate.

Real data comes from NOAA GHCN-hourly (GHCNh), one pipe-separated file per
station-year:
  https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/<year>/psv/GHCNh_<station>_<year>.psv
GHCNh replaced ISD, the hourly dataset NCEI used to compute GSOD. GSOD itself
was retired on 2025-08-29, so this module computes the same daily summaries
from the hourly reports and converts them to GSOD's units (degrees F, mb,
knots, inches). That keeps the model's features identical to the KFP pipeline.

Synthetic data is a deterministic offline stand-in for development and drills.
"""
import io
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config, store

COLUMNS = ["date", "temp", "tmax", "tmin", "dewp", "slp", "wdsp", "prcp"]
HOURLY_VARS = ["temperature", "dew_point_temperature", "sea_level_pressure",
               "wind_speed", "precipitation"]
MS_TO_KNOTS = 1.943844
SENTINELS = {9999.9: np.nan, 999.9: np.nan, 99.99: np.nan}   # GSOD "no report" values


class DataValidationError(ValueError):
    """Bad data. Retrying will not fix it, so tasks treat this as a hard failure."""


class InsufficientHistory(ValueError):
    """Not enough complete recent days to build features (a gap in the source).
    Expected and temporary, so tasks skip rather than fail."""


class SourceUnavailable(RuntimeError):
    """The upstream source is missing or has stopped updating (a retired dataset,
    a moved URL). Different from "not published yet": waiting will not help."""


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
    for g in config.DRILL_GAP_DATES:                  # gap drill, see config.py
        out.loc[out.date == date.fromisoformat(g), COLUMNS[1:]] = np.nan
    return out


# ---------------------------------------------------------------- GHCNh
def summarize_hourly(h: pd.DataFrame) -> pd.DataFrame:
    """Hourly GHCNh rows -> one row per UTC day, in GSOD units.

    tmax/tmin/temp  max/min/mean of hourly temperature (C -> F)
    dewp            mean dew point (C -> F)
    slp             mean sea-level pressure (hPa = mb)
    wdsp            mean wind speed (m/s -> knots)
    prcp            sum of precipitation in routine hourly reports (mm -> inches)
    hours           distinct UTC hours with a temperature report
    Values whose quality code is in config.REJECT_QC are dropped first. Days with
    fewer than MIN_HOURS_PER_DAY hours are blanked (incomplete, e.g. the newest day).
    """
    h = h.copy()
    t = pd.to_datetime(h["DATE"], utc=True)
    h["day"], h["hour"] = t.dt.date, t.dt.hour
    rejected = {}
    for v in HOURLY_VARS:
        h[v] = pd.to_numeric(h[v], errors="coerce")
        qc = f"{v}_Quality_Code"
        if qc in h.columns:
            bad = h[qc].astype(str).str.strip().isin(config.REJECT_QC)
            rejected[v] = int((bad & h[v].notna()).sum())
            h.loc[bad, v] = np.nan
    routine = pd.to_numeric(h["Minute"], errors="coerce") == config.ROUTINE_MINUTE
    g = h.groupby("day")
    d = pd.DataFrame({
        "tmax": g["temperature"].max(), "tmin": g["temperature"].min(),
        "temp": g["temperature"].mean(), "dewp": g["dew_point_temperature"].mean(),
        "slp": g["sea_level_pressure"].mean(), "wdsp": g["wind_speed"].mean(),
        "prcp": h[routine].groupby("day")["precipitation"].sum(min_count=0),
        "hours": h[h["temperature"].notna()].groupby("day")["hour"].nunique(),
    })
    d["hours"] = d["hours"].fillna(0).astype(int)
    d["prcp"] = d["prcp"].fillna(0.0)
    for c in ("tmax", "tmin", "temp", "dewp"):
        d[c] = d[c] * 9 / 5 + 32
    d["wdsp"] = d["wdsp"] * MS_TO_KNOTS
    d["prcp"] = d["prcp"] / 25.4
    incomplete = d["hours"] < config.MIN_HOURS_PER_DAY
    d.loc[incomplete, COLUMNS[1:]] = np.nan
    d = d.rename_axis("date").reset_index()
    d.attrs["rejected_by_qc"] = rejected
    return d[COLUMNS + ["hours"]]


def _ghcnh_year(year: int, now: datetime) -> pd.DataFrame:
    import requests   # imported here so offline/synthetic use needs no network stack
    url = config.GHCNH_URL.format(year=year, station=config.STATION_ID)
    r = requests.get(url, timeout=300)
    if r.status_code == 404:
        if year > now.year or (year == now.year and now.timetuple().tm_yday <= 7):
            return pd.DataFrame(columns=COLUMNS + ["hours"])   # year file not started yet
        raise SourceUnavailable(f"{url} returned 404: the source file is missing "
                                "(dataset moved or retired?)")
    r.raise_for_status()                              # 5xx etc. -> exception -> Airflow retries
    cols = ["DATE", "Minute"] + HOURLY_VARS + [f"{v}_Quality_Code" for v in HOURLY_VARS]
    h = pd.read_csv(io.StringIO(r.text), sep="|", usecols=lambda c: c in cols,
                    dtype=str, low_memory=False)
    missing = {"DATE", "Minute", "temperature"} - set(h.columns)
    if missing:
        raise SourceUnavailable(f"{url} is missing columns {sorted(missing)} (format changed?)")
    for v in HOURLY_VARS:
        if v not in h.columns:
            h[v] = np.nan
    return summarize_hourly(h)


# ---------------------------------------------------------------- cache
def _cache_path(year: int):
    return config.DATA_DIR / "raw" / f"daily_{config.STATION_ID}_{year}_{config.SOURCE}.csv"


def fetch_year(year: int, force: bool = False, now: datetime | None = None,
               max_age_hours: float | None = None) -> pd.DataFrame:
    """Return one station-year of daily rows, using a local cache of the summary.

    Past years are downloaded once. A year that is still growing is
    re-downloaded when its cache is older than max_age_hours (default
    REFRESH_HOURS) or force=True, because NOAA keeps appending to it.
    """
    now = now or datetime.now(timezone.utc)
    path = _cache_path(year)
    max_age = config.REFRESH_HOURS if max_age_hours is None else max_age_hours
    still_growing = year >= now.year - (1 if now.month == 1 else 0)
    stale = (not path.exists()
             or (still_growing and time.time() - path.stat().st_mtime > max_age * 3600))
    if force or stale:
        df = _synthetic_year(year, now) if config.SOURCE == "synthetic" else _ghcnh_year(year, now)
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


def observation_status(d: date, now: datetime | None = None) -> dict:
    """Is day d complete in the source yet?

    status "available": d is complete (enough hourly reports).
    status "pending":   d is not complete yet and nothing much newer is either;
                        keep waiting (GHCNh lags about 3 days).
    status "missing":   d is incomplete but a day GAP_CONFIRM_DAYS or more
                        later is already complete. The source has a gap; d is
                        not coming, and waiting would only block later days.
    Raises SourceUnavailable if the source looks retired or stalled: its newest
    complete day is more than MAX_SOURCE_LAG_DAYS behind the clock. Without
    this check a dead feed looks exactly like "not published yet", and a
    sensor would simply wait until it timed out.
    """
    now = now or datetime.now(timezone.utc)
    df = fetch_year(d.year, now=now, max_age_hours=1)
    complete = df[df.tmax.notna()]
    newest = complete.date.max() if len(complete) else None
    if d.year == now.date().year and (
            newest is None or (now.date() - newest).days > config.MAX_SOURCE_LAG_DAYS):
        raise SourceUnavailable(f"newest complete day in {config.SOURCE} is {newest}; "
                                f"more than {config.MAX_SOURCE_LAG_DAYS} days behind {now.date()}")
    row = df[df.date == d]
    hours = int(row["hours"].iloc[0]) if len(row) and "hours" in row and pd.notna(row["hours"].iloc[0]) else None
    if (complete.date == d).any():
        status = "available"
    elif newest is not None and (newest - d).days >= config.GAP_CONFIRM_DAYS:
        status = "missing"
    else:
        status = "pending"
    return {"date": d.isoformat(), "status": status, "hours": hours,
            "newest_complete": newest.isoformat() if newest else None}


def observation_available(d: date, now: datetime | None = None) -> bool:
    return observation_status(d, now)["status"] == "available"


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
    days_present = df[df.tmax.notna()].date.nunique()
    if days_present < window_days * 0.7:
        problems.append(f"only {days_present}/{window_days} days present in the window")
    if problems:
        raise DataValidationError("; ".join(problems))
    return {"date": d.isoformat(), "rows": int(len(df)), "days_present": int(days_present),
            "tmax": float(today.tmax.iloc[0])}
