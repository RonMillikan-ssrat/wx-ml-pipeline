"""Settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


# Station: Houston Intercontinental (IAH). NCEI names files USAF+WBAN.
STATION_ID = _env("WX_STATION_ID", "72243012960")

# "ncei" = real GSOD CSVs from NCEI over HTTPS (no GCP credentials needed)
# "synthetic" = deterministic offline weather, for development and drills
SOURCE = _env("WX_SOURCE", "ncei")
NCEI_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{station}.csv"

# Synthetic data only: pretend observations publish this many days late, so the
# sensor has something to wait for (real GSOD typically lags a few days).
SYNTH_LAG_DAYS = _env("WX_SYNTH_LAG_DAYS", 2, int)

# Synthetic data only, for failure drills: report an impossible Tmax (150 F)
# on this date (YYYY-MM-DD) so validation has something to reject.
DRILL_BAD_OBS_DATE = _env("WX_DRILL_BAD_OBS_DATE", "")

DATA_DIR = Path(_env("WX_DATA_DIR", "./data"))

# Current-year file is re-downloaded when the cached copy is older than this.
REFRESH_HOURS = _env("WX_REFRESH_HOURS", 6, float)

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
