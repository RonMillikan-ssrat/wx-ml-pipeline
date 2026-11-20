"""Rollback drill: put a deliberately broken model into production.

The model is trained on shuffled targets, so it has learned nothing useful
and will lose to persistence. It is registered and promoted the normal way
(@previous keeps the good model), exactly as a bad release would arrive.
Then let wx_daily catch it:

    python scripts/drill_bad_model.py
    # reprocess enough days for ROLLBACK_MIN_N verifications, allowing actions:
    airflow backfill create --dag-id wx_daily --from-date <D1> --to-date <D2> \
        --reprocess-behavior completed --max-active-runs 1 \
        --dag-run-conf '{"allow_actions": true}'

Run inside the Airflow container (docker compose exec airflow-scheduler ...)
or anywhere with MLFLOW_TRACKING_URI, WX_SOURCE and WX_DATA_DIR set.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wxml import config, data, model, registry  # noqa: E402
from wxml.features import build_features  # noqa: E402


def main():
    good = registry.alias_version(config.PROD_ALIAS)
    if good is None:
        sys.exit("no @production model yet; run wx_retrain first")
    f = build_features(data.load_range(date(config.TRAIN_START_YEAR, 1, 1), date(2025, 12, 31)))
    f["y"] = np.random.default_rng(42).permutation(f["y"].to_numpy())   # destroy the signal
    bad = model.train(f)
    version = registry.register(
        bad, metrics={"skill": -1.0},
        params={"drill": "shuffled targets"},
        tags={"airflow_run_id": "drill", "train_end": "2025-12-31", "drill": "true",
              "status": "candidate"})
    out = registry.promote(version, reason="ROLLBACK DRILL: deliberately broken model")
    print(f"promoted drill model v{version}; @previous is now v{out['previous']} (was production)")


if __name__ == "__main__":
    main()
