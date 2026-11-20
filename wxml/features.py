"""Feature engineering shared by training and daily inference.

build_features() reproduces the body of make_features() in pipeline.py line for
line. KFP lightweight components must be self-contained, so the code cannot be
imported there; tests/test_feature_parity.py runs both and checks they match.
If you change one, change the other, or the test fails. A mismatch between
training-time and serving-time features ("training/serving skew") is one of
the most common ways a model that scored well in research goes wrong in
production.
"""
import numpy as np
import pandas as pd

FEATURES = ["tmax_l0", "tmax_l1", "tmax_l2", "tmin_l0", "dewp_l0", "slp_l0", "dslp",
            "wdsp_l0", "prcp_l0", "doy_sin", "doy_cos"]


def build_features(raw: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    """raw has a 'date' column plus GSOD fields. Returns features indexed by date.

    with_target=True:  adds y = tomorrow's tmax and drops incomplete rows (training).
    with_target=False: no target, so the most recent day survives (inference).
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").asfreq("D")
    # ---- identical to pipeline.make_features from here ----
    df = df.interpolate(limit=2)                       # fill short gaps only
    f = pd.DataFrame(index=df.index)
    for k in (0, 1, 2):
        f[f"tmax_l{k}"] = df.tmax.shift(k)
    f["tmin_l0"], f["dewp_l0"], f["slp_l0"] = df.tmin, df.dewp, df.slp
    f["dslp"] = df.slp.diff()                          # pressure tendency
    f["wdsp_l0"], f["prcp_l0"] = df.wdsp, df.prcp
    doy = df.index.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    if with_target:
        f["y"] = df.tmax.shift(-1)                     # target
    # ---- end identical section ----
    return f.dropna()
