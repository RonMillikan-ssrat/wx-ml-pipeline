"""Training/serving skew guard.

Runs the actual make_features component function from pipeline.py (the code
GEAP executes) and wxml.features.build_features on the same raw data, and
requires identical output. Needs kfp installed; skipped otherwise.
"""
import pandas as pd
import pytest

kfp = pytest.importorskip("kfp")

from wxml import data  # noqa: E402
from wxml.features import FEATURES, build_features  # noqa: E402


def test_build_features_matches_kfp_component(tmp_path, monkeypatch):
    from kfp.dsl import Dataset

    import pipeline

    raw = data._synthetic_series().iloc[:1200].copy()
    raw.loc[[100, 101, 500], "tmax"] = float("nan")       # gaps the interpolation fills
    raw.loc[[800, 801, 802], "slp"] = float("nan")        # 3-day gap: longer than limit=2
    raw_csv = tmp_path / "raw.csv"
    raw.to_csv(raw_csv, index=False)

    def ds(p):
        a = Dataset(name=p.stem, uri=str(p))
        a.path = str(p)
        return a

    test_start = "2007-06-01"
    pipeline.make_features.python_func(raw=ds(raw_csv), test_start=test_start,
                                       train=ds(tmp_path / "train.csv"),
                                       test=ds(tmp_path / "test.csv"))
    kfp_all = pd.concat([pd.read_csv(tmp_path / "train.csv", index_col=0, parse_dates=True),
                         pd.read_csv(tmp_path / "test.csv", index_col=0, parse_dates=True)])

    ours = build_features(raw)
    assert list(ours.columns) == FEATURES + ["y"]
    pd.testing.assert_frame_equal(ours, kfp_all, check_freq=False, check_names=False)
