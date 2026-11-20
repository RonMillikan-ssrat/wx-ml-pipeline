"""Settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


# Station: Houston Intercontinental (IAH), GHCN identifier.
STATION_ID = _env("WX_STATION_ID", "USW00012960")

# "ghcnh"     = NOAA GHCN-hourly observations, summarized to daily values here.
#               GHCNh replaced ISD, the hourly dataset GSOD was computed from.
#               (GSOD itself was retired by NCEI on 2025-08-29.)
# "synthetic" = deterministic offline weather, for development and drills
SOURCE = _env("WX_SOURCE", "ghcnh")
GHCNH_URL = ("https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/"
             "access/by-year/{year}/psv/GHCNh_{station}_{year}.psv")

# A day counts as observed only with this many distinct UTC hours of temperature.
MIN_HOURS_PER_DAY = _env("WX_MIN_HOURS_PER_DAY", 18, int)

# GHCNh flags each value with a quality code. Values whose code is in this
# comma-separated list are dropped before the daily summary. Empty = keep all;
# set it after looking at which codes the station actually reports
# (scripts/validate_ghcnh.py prints them).
REJECT_QC = {c.strip() for c in _env("WX_REJECT_QC", "").split(",") if c.strip()}

# Precipitation is summed over routine hourly reports only (ASOS METARs at
# minute :53 at IAH); special reports repeat the accumulation since the last
# routine report and would double-count.
ROUTINE_MINUTE = _env("WX_ROUTINE_MINUTE", 53, int)

# The source is "stale" (the sensor fails instead of waiting) when its newest
# observation is older than this. Catches a retired or stalled upstream feed.
MAX_SOURCE_LAG_DAYS = _env("WX_MAX_SOURCE_LAG_DAYS", 10, int)

# Synthetic data only: pretend observations publish this many days late, so the
# sensor has something to wait for (real GHCNh lags about 3 days).
SYNTH_LAG_DAYS = _env("WX_SYNTH_LAG_DAYS", 2, int)

# Synthetic data only, for failure drills: report an impossible Tmax (150 F)
# on this date (YYYY-MM-DD) so validation has something to reject.
DRILL_BAD_OBS_DATE = _env("WX_DRILL_BAD_OBS_DATE", "")

DATA_DIR = Path(_env("WX_DATA_DIR", "./data"))

# A still-growing year is re-downloaded when the cached copy is older than this
# (about 35 MB per station-year, so not on every sensor poke).
REFRESH_HOURS = _env("WX_REFRESH_HOURS", 2, float)

# Training (same model and hyperparameters as the KFP pipeline defaults)
TRAIN_START_YEAR = _env("WX_TRAIN_START_YEAR", 2010, int)
HOLDOUT_DAYS = _env("WX_HOLDOUT_DAYS", 90, int)
MAX_ITER = _env("WX_MAX_ITER", 300, int)
LEARNING_RATE = _env("WX_LEARNING_RATE", 0.05, float)

# Gates
SKILL_GATE = _env("WX_SKILL_GATE", 0.05, float)           # promote only above this skill
ROLLBACK_SKILL = _env("WX_ROLLBACK_SKILL", 0.0, float)    # roll back below this rolling skill
ROLLBACK_WINDOW_DAYS = _env("WX_ROLLBACK_WINDOW_DAYS", 30, int)
ROLLBACK_MIN_N = _env("WX_ROLLBACK_MIN_N", 14, int)       # need this many verified days first

# MLflow Model Registry
MLFLOW_TRACKING_URI = _env("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = _env("WX_MODEL_NAME", "wx-tmax-iah")
EXPERIMENT = _env("WX_EXPERIMENT", "wx-tmax-iah")
PROD_ALIAS = "production"
PREV_ALIAS = "previous"

# Optional Slack-compatible webhook for failure alerts
ALERT_WEBHOOK = _env("WX_ALERT_WEBHOOK", "")
