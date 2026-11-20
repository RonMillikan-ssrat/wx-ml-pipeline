"""Check the GHCNh daily summaries against NOAA's own GSOD for the same days.

GSOD was retired on 2025-08-29, but its 2025 file (through late August) is
still online. Both are derived from the same hourly reports, so for the
overlapping days they should agree closely. This prints, per variable:
days compared, mean difference (GHCNh - GSOD), mean absolute difference,
and correlation. It also lists the quality codes present in the hourly file.

    docker compose exec airflow-scheduler python /opt/wx/scripts/validate_ghcnh.py
    docker compose exec airflow-scheduler python /opt/wx/scripts/validate_ghcnh.py --year 2024
"""
import argparse
import io
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wxml import config, data  # noqa: E402

GSOD_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/72243012960.csv"
GSOD_COLS = {"DATE": "date", "TEMP": "temp", "MAX": "tmax", "MIN": "tmin", "DEWP": "dewp",
             "SLP": "slp", "WDSP": "wdsp", "PRCP": "prcp"}


def gsod(year):
    r = requests.get(GSOD_URL.format(year=year), timeout=120)
    r.raise_for_status()
    g = pd.read_csv(io.StringIO(r.text), dtype=str, skipinitialspace=True)[list(GSOD_COLS)]
    g = g.rename(columns=GSOD_COLS)
    g["date"] = pd.to_datetime(g["date"].str.strip()).dt.date
    for c in list(GSOD_COLS.values())[1:]:
        g[c] = pd.to_numeric(g[c].str.strip(), errors="coerce")
    return g.replace({9999.9: np.nan, 999.9: np.nan, 99.99: np.nan})


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)    # corr of constant columns
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    year = ap.parse_args().year

    url = config.GHCNH_URL.format(year=year, station=config.STATION_ID)
    print(f"GHCNh: {url}")
    raw = requests.get(url, timeout=300).text
    qc_cols = [f"{v}_Quality_Code" for v in data.HOURLY_VARS]
    h = pd.read_csv(io.StringIO(raw), sep="|", dtype=str, low_memory=False,
                    usecols=lambda c: c in ["DATE", "Minute"] + data.HOURLY_VARS + qc_cols)
    print(f"hourly rows: {len(h)}   report minutes (top 5): "
          f"{h['Minute'].value_counts().head(5).to_dict()}")
    print("\nquality codes present (value: count of non-missing values carrying it):")
    for v in data.HOURLY_VARS:
        qc = f"{v}_Quality_Code"
        if qc in h.columns:
            has_val = pd.to_numeric(h[v], errors="coerce").notna()
            counts = h.loc[has_val, qc].fillna("(blank)").str.strip().replace("", "(blank)")
            print(f"  {v:<22} {counts.value_counts().head(8).to_dict()}")

    ours = data.summarize_hourly(h)
    g = gsod(year)
    m = ours.merge(g, on="date", suffixes=("_ghcnh", "_gsod"))
    print(f"\n{len(m)} overlapping days ({m.date.min()} .. {m.date.max()}); "
          f"REJECT_QC={sorted(config.REJECT_QC) or 'none'}\n")
    print(f"{'var':<6}{'n':>5}{'mean diff':>11}{'mean |diff|':>13}{'corr':>8}   units")
    units = {"tmax": "F", "tmin": "F", "temp": "F", "dewp": "F", "slp": "mb", "wdsp": "kt", "prcp": "in"}
    for v in ["tmax", "tmin", "temp", "dewp", "slp", "wdsp", "prcp"]:
        a, b = m[f"{v}_ghcnh"], m[f"{v}_gsod"]
        ok = a.notna() & b.notna()
        diff = (a - b)[ok]
        corr = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else float("nan")
        print(f"{v:<6}{ok.sum():>5}{diff.mean():>11.2f}{diff.abs().mean():>13.2f}{corr:>8.3f}   {units[v]}")
    worst = m.assign(d=(m.tmax_ghcnh - m.tmax_gsod).abs()).nlargest(5, "d")
    print("\nlargest tmax differences:")
    print(worst[["date", "tmax_ghcnh", "tmax_gsod", "hours"]].to_string(index=False))
    print(f"\nchecked at {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
