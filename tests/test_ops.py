"""Idempotency, validation, monitoring and rollback logic. No server needed:
MLflow runs against a local SQLite file, data is synthetic."""
from datetime import date, timedelta

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from wxml import config
    monkeypatch.setattr(config, "SOURCE", "synthetic")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(config, "ROLLBACK_MIN_N", 5)
    monkeypatch.chdir(tmp_path)          # MLflow writes ./mlruns artifacts here, not in the repo
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    return config


def test_store_overwrite_is_idempotent(env):
    from wxml import store
    p = store.forecast_path(date(2026, 1, 2))
    store.write_json(p, {"a": 1})
    store.write_json(p, {"a": 2})
    assert store.read_json(p) == {"a": 2}
    assert list(p.parent.iterdir()) == [p]            # no leftover temp files


def test_validation_rejects_bad_data(env):
    from wxml import data
    d = date(2026, 3, 10)
    df = data.load_range(d - timedelta(days=13), d)
    data.validate_window(df, d, 14)                   # clean data passes
    bad = df.copy()
    bad.loc[bad.index[-1], "tmax"] = 150.0
    with pytest.raises(data.DataValidationError, match="out-of-range"):
        data.validate_window(bad, d, 14)
    with pytest.raises(data.DataValidationError, match="no row"):
        data.validate_window(df[df.date < d], d, 14)


def test_synthetic_publication_lag(env, monkeypatch):
    from datetime import datetime, timezone

    from wxml import data
    monkeypatch.setattr(env, "SYNTH_LAG_DAYS", 2)
    now = datetime(2026, 5, 20, 12, tzinfo=timezone.utc)
    df = data._synthetic_year(2026, now)
    assert df.date.max() == date(2026, 5, 18)


def _train_and_promote(end, run_id):
    from wxml import ops, registry
    r = ops.train_candidate(end, run_id)
    v = ops.register_candidate(r, run_id)
    assert ops.register_candidate(r, run_id) == v     # retry reuses the version
    registry.promote(v, "test")
    return v


def test_daily_cycle_and_rollback(env):
    import joblib
    import numpy as np

    from wxml import ops, registry

    v1 = _train_and_promote(date(2026, 6, 1), "train-1")
    v2 = _train_and_promote(date(2026, 6, 8), "train-2")
    assert registry.alias_version("production") == v2
    assert registry.alias_version("previous") == v1

    # Sabotage v2's behaviour: register a model trained on shuffled targets and promote it.
    from wxml import data, model
    from wxml.features import build_features
    f = build_features(data.load_range(date(2015, 1, 1), date(2026, 1, 1)))
    f["y"] = np.random.default_rng(1).permutation(f["y"].to_numpy())
    bad = model.train(f)
    out = ops.candidate_dir("bad")
    out.mkdir(parents=True)
    joblib.dump(bad, out / "model.joblib")
    res = {"candidate": {"skill": -1.0, "rmse_F": 9.0, "rmse_persistence_F": 4.0, "n_test": 1},
           "train_end": "2026-01-01", "holdout_end": "2026-01-01", "model_path": str(out / "model.joblib")}
    v_bad = ops.register_candidate(res, "bad")
    registry.promote(v_bad, "drill")

    start = date(2026, 7, 1)
    for i in range(8):
        d = start + timedelta(days=i)
        ops.validate_day(d)
        ops.verify_day(d)
        ops.forecast_day(d, f"run-{i}")
    stats = ops.check_skill(start + timedelta(days=7))
    assert stats["model_version"] == v_bad and stats["n"] == 7
    assert stats["degraded"], stats

    first = registry.rollback(v_bad, "test")
    assert first["changed"] and first["production"] == v2
    again = registry.rollback(v_bad, "test")             # retry: no second rollback
    assert not again["changed"] and registry.alias_version("production") == v2
    assert registry.version_tags(v_bad)["status"] == "rolled_back"


def test_gate(env):
    from wxml import ops
    good = {"candidate": {"skill": 0.2, "rmse_F": 3.0}, "production": {"rmse_F": 3.1},
            "production_version": "1"}
    assert ops.gate(good)[0]
    worse = {**good, "production": {"rmse_F": 2.9}}
    assert not ops.gate(worse)[0]
    weak = {**good, "candidate": {"skill": 0.01, "rmse_F": 3.0}}
    assert not ops.gate(weak)[0]
