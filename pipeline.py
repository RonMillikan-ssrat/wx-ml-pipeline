"""
GEAP + Kubeflow task: next-day Tmax forecast for Houston (IAH) from NOAA GSOD.

DAG:
  extract_gsod -> make_features -> train -> evaluate -> [skill > threshold] -> register_model

Compile:   python pipeline.py            -> wx_tmax_pipeline.yaml (KFP IR YAML)
The same YAML runs on: kfp local (run_local.py), GEAP Pipelines (run_geap.py),
or an OSS Kubeflow Pipelines cluster (upload via KFP UI / kfp.Client).
"""
import os

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

PROJECT = os.environ.get("WX_PROJECT", "wx-ml-pipeline-0")
BASE = "python:3.11-slim"
SKLEARN = "scikit-learn==1.5.2"   # must match the serving container's sklearn minor version


# ---------------------------------------------------------------- 1. extract
@dsl.component(base_image=BASE,
               packages_to_install=["pandas==2.2.3", "numpy<2.3",
                                    "google-cloud-bigquery==3.27.0", "db-dtypes", "pyarrow"])
def extract_gsod(project: str, stn: str, wban: str, start_year: int, end_year: int,
                 source: str, raw: Output[Dataset]):
    """Pull daily GSOD rows for one station. source='bq' or 'synthetic' (offline testing)."""
    import numpy as np
    import pandas as pd

    if source == "bq":
        from google.cloud import bigquery
        sql = f"""
          SELECT DATE(CAST(year AS INT64), CAST(mo AS INT64), CAST(da AS INT64)) AS date,
                 temp, max AS tmax, min AS tmin, dewp, slp, wdsp, prcp
          FROM `bigquery-public-data.noaa_gsod.gsod*`
          WHERE _TABLE_SUFFIX BETWEEN '{start_year}' AND '{end_year}'
            AND stn = '{stn}' AND wban = '{wban}'
          ORDER BY date"""
        df = bigquery.Client(project=project).query(sql).to_dataframe()
        # GSOD missing-value sentinels
        df = df.replace({9999.9: np.nan, 999.9: np.nan, 99.99: np.nan})
    else:
        rng = np.random.default_rng(0)
        date = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        doy = date.dayofyear.to_numpy()
        seas = 80 + 15 * np.sin(2 * np.pi * (doy - 110) / 365.25)
        anom = np.zeros(len(date))
        for i in range(1, len(date)):                      # AR(1) weather noise
            anom[i] = 0.7 * anom[i - 1] + rng.normal(0, 4)
        tmax = seas + anom
        df = pd.DataFrame({"date": date, "tmax": tmax, "tmin": tmax - 18 + rng.normal(0, 2, len(date)),
                           "temp": tmax - 9, "dewp": tmax - 20 + rng.normal(0, 3, len(date)),
                           "slp": 1015 - 0.3 * anom + rng.normal(0, 2, len(date)),
                           "wdsp": rng.gamma(2, 3, len(date)), "prcp": rng.exponential(0.1, len(date))})
    print(f"rows={len(df)}  range={df.date.min()}..{df.date.max()}")
    print(df.isna().mean().round(3).to_string())
    df.to_csv(raw.path, index=False)


# ---------------------------------------------------------------- 2. features
@dsl.component(base_image=BASE, packages_to_install=["pandas==2.2.3", "numpy<2.3"])
def make_features(raw: Input[Dataset], test_start: str,
                  train: Output[Dataset], test: Output[Dataset]) -> int:
    """Lag features, target = tomorrow's Tmax. Time-based split (no shuffling -> no leakage)."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(raw.path, parse_dates=["date"]).set_index("date").asfreq("D")
    df = df.interpolate(limit=2)                       # fill short gaps only
    f = pd.DataFrame(index=df.index)
    for k in (0, 1, 2):
        f[f"tmax_l{k}"] = df.tmax.shift(k)
    f["tmin_l0"], f["dewp_l0"], f["slp_l0"] = df.tmin, df.dewp, df.slp
    f["dslp"] = df.slp.diff()                          # pressure tendency
    f["wdsp_l0"], f["prcp_l0"] = df.wdsp, df.prcp
    doy = df.index.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["y"] = df.tmax.shift(-1)                         # target
    f = f.dropna()
    tr, te = f[f.index < test_start], f[f.index >= test_start]
    print(f"train={len(tr)}  test={len(te)}  features={f.shape[1] - 1}")
    tr.to_csv(train.path)
    te.to_csv(test.path)
    return len(tr)


# ---------------------------------------------------------------- 3. train
@dsl.component(base_image=BASE, packages_to_install=["pandas==2.2.3", SKLEARN, "joblib"])
def train(train: Input[Dataset], max_iter: int, learning_rate: float, model: Output[Model]):
    import os
    import joblib
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor

    d = pd.read_csv(train.path, index_col=0)
    X, y = d.drop(columns="y"), d.y
    m = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate,
                                      random_state=0).fit(X, y)
    # Model Registry's prebuilt sklearn server expects <dir>/model.joblib
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(m, os.path.join(model.path, "model.joblib"))
    model.metadata["framework"] = "sklearn"
    model.metadata["features"] = list(X.columns)
    print("saved", model.path, "| model.uri =", model.uri)


# ---------------------------------------------------------------- 4. evaluate
@dsl.component(base_image=BASE, packages_to_install=["pandas==2.2.3", SKLEARN, "joblib"])
def evaluate(test: Input[Dataset], model: Input[Model], metrics: Output[Metrics]) -> float:
    """Returns skill vs persistence (tomorrow = today): 1 - RMSE_model / RMSE_persist."""
    import os
    import joblib
    import numpy as np
    import pandas as pd

    d = pd.read_csv(test.path, index_col=0)
    m = joblib.load(os.path.join(model.path, "model.joblib"))
    pred = m.predict(d.drop(columns="y"))
    rmse = float(np.sqrt(np.mean((pred - d.y) ** 2)))
    rmse_p = float(np.sqrt(np.mean((d.tmax_l0 - d.y) ** 2)))
    skill = 1 - rmse / rmse_p
    for k, v in {"rmse_F": rmse, "rmse_persistence_F": rmse_p, "skill": skill,
                 "n_test": float(len(d))}.items():
        metrics.log_metric(k, round(v, 4))
    print(f"RMSE={rmse:.2f}F  persistence={rmse_p:.2f}F  skill={skill:.3f}")
    return skill


# ---------------------------------------------------------------- 5. register
@dsl.component(base_image=BASE, packages_to_install=["google-cloud-aiplatform>=1.70"])
def register_model(project: str, location: str, model: Input[Model],
                   serving_image: str, skill: float) -> str:
    """Upload artifact dir to GEAP Model Registry (no endpoint -> no idle cost)."""
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=location)
    m = aiplatform.Model.upload(
        display_name="wx-tmax-iah",
        artifact_uri=model.uri,                 # gs://.../model  (dir containing model.joblib)
        serving_container_image_uri=serving_image,
        labels={"skill_x1000": str(int(skill * 1000))},
    )
    print("registered", m.resource_name)
    return m.resource_name


# ---------------------------------------------------------------- pipeline
@dsl.pipeline(name="wx-tmax-iah", description="GSOD next-day Tmax, Houston IAH")
def wx_pipeline(project: str = PROJECT,
                location: str = "us-central1",
                stn: str = "722430", wban: str = "12960",      # Houston Intercontinental
                start_year: int = 2010, end_year: int = 2025,
                test_start: str = "2024-01-01",
                source: str = "bq",
                max_iter: int = 300, learning_rate: float = 0.05,
                skill_threshold: float = 0.05,
                register: bool = True,
                serving_image: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"):
    ex = extract_gsod(project=project, stn=stn, wban=wban, start_year=start_year,
                      end_year=end_year, source=source)
    ft = make_features(raw=ex.outputs["raw"], test_start=test_start)
    tr = train(train=ft.outputs["train"], max_iter=max_iter, learning_rate=learning_rate)
    tr.set_cpu_limit("2").set_memory_limit("4G")   # GEAP picks a machine that fits
    ev = evaluate(test=ft.outputs["test"], model=tr.outputs["model"])
    with dsl.If(register == True, name="register-enabled"):      # noqa: E712 (KFP DSL)
        with dsl.If(ev.outputs["Output"] > skill_threshold, name="skill-gate"):
            register_model(project=project, location=location, model=tr.outputs["model"],
                           serving_image=serving_image, skill=ev.outputs["Output"])


if __name__ == "__main__":
    compiler.Compiler().compile(wx_pipeline, "wx_tmax_pipeline.yaml")
    print("wrote wx_tmax_pipeline.yaml")
