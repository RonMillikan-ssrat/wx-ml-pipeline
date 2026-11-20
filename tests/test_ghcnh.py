"""GHCNh hourly -> daily summaries, units, and failure modes of the source."""
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from wxml import config, data


def hourly(day="2026-07-01", hours=range(24), temp=lambda h: 25 + 5 * np.sin(np.pi * h / 24),
           qc=None, minutes=(53,)):
    rows = []
    for h in hours:
        for m in minutes:
            rows.append({"DATE": f"{day}T{h:02d}:{m:02d}:00", "Minute": str(m),
                         "temperature": str(round(temp(h), 1)), "dew_point_temperature": "20.0",
                         "sea_level_pressure": "1015.0", "wind_speed": "5.0",
                         "precipitation": "1.0" if h < 2 else "", "temperature_Quality_Code": ""})
    df = pd.DataFrame(rows)
    if qc:
        for i, code in qc.items():
            df.loc[i, "temperature_Quality_Code"] = code
    return df


@pytest.fixture(autouse=True)
def cfg(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MIN_HOURS_PER_DAY", 18)
    monkeypatch.setattr(config, "ROUTINE_MINUTE", 53)
    monkeypatch.setattr(config, "REJECT_QC", set())
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)


def test_daily_summary_and_units():
    d = data.summarize_hourly(hourly()).iloc[0]
    assert d.date == date(2026, 7, 1) and d.hours == 24
    assert d.tmax == pytest.approx(30 * 9 / 5 + 32, abs=0.2)       # 30 C -> 86 F
    assert d.tmin == pytest.approx(25 * 9 / 5 + 32, abs=0.2)
    assert d.dewp == pytest.approx(68.0)                             # 20 C
    assert d.slp == pytest.approx(1015.0)                            # hPa = mb
    assert d.wdsp == pytest.approx(5 * 1.943844)                     # m/s -> knots
    assert d.prcp == pytest.approx(2.0 / 25.4)                       # two hours of 1 mm


def test_specials_do_not_double_count_precip():
    h = hourly(minutes=(53, 10))          # a special report at :10 every hour
    assert data.summarize_hourly(h).iloc[0].prcp == pytest.approx(2.0 / 25.4)


def test_incomplete_day_is_blanked():
    d = data.summarize_hourly(hourly(hours=range(12))).iloc[0]
    assert d.hours == 12 and np.isnan(d.tmax) and np.isnan(d.slp)


def test_quality_codes_rejected_only_when_configured(monkeypatch):
    h = hourly(temp=lambda hr: 25.0, qc={5: "X"})
    h.loc[5, "temperature"] = "60.0"                 # a flagged spike
    assert data.summarize_hourly(h).iloc[0].tmax == pytest.approx(140.0)
    monkeypatch.setattr(config, "REJECT_QC", {"X"})
    assert data.summarize_hourly(h).iloc[0].tmax == pytest.approx(77.0)


class Resp:
    def __init__(self, code, text=""):
        self.status_code, self.text = code, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_404_is_loud(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(404))
    now = datetime(2026, 11, 20, tzinfo=timezone.utc)
    with pytest.raises(data.SourceUnavailable):
        data._ghcnh_year(2026, now)                  # current year, mid-November
    assert data._ghcnh_year(2027, now).empty         # future year: fine, not started


def test_stale_source_is_detected(monkeypatch):
    import requests
    two_days = pd.concat([hourly("2026-11-01"), hourly("2026-11-02")])
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: Resp(200, two_days.to_csv(sep="|", index=False)))
    monkeypatch.setattr(config, "SOURCE", "ghcnh")
    fresh = datetime(2026, 11, 5, tzinfo=timezone.utc)
    assert data.observation_available(date(2026, 11, 2), now=fresh)
    assert not data.observation_available(date(2026, 11, 3), now=fresh)
    later = datetime(2026, 11, 20, tzinfo=timezone.utc)      # 18 days with nothing new
    with pytest.raises(data.SourceUnavailable, match="behind"):
        data.observation_available(date(2026, 11, 3), now=later)


def test_gap_vs_pending(monkeypatch):
    """Reproduces IAH Nov 2026: Nov 9-12 incomplete, Nov 13-16 complete, Nov 17 partial."""
    import requests
    days = [hourly(f"2026-11-{d:02d}") for d in (6, 7, 8, 13, 14, 15, 16)]
    days += [hourly("2026-11-09", hours=range(16)), hourly("2026-11-12", hours=range(3)),
             hourly("2026-11-17", hours=range(16))]
    psv = pd.concat(days).to_csv(sep="|", index=False)
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(200, psv))
    monkeypatch.setattr(config, "SOURCE", "ghcnh")
    monkeypatch.setattr(config, "GAP_CONFIRM_DAYS", 2)
    now = datetime(2026, 11, 20, tzinfo=timezone.utc)
    st = lambda d: data.observation_status(date(2026, 11, d), now=now)["status"]  # noqa: E731
    assert st(8) == "available"
    assert [st(d) for d in (9, 10, 11, 12)] == ["missing"] * 4    # newest complete is the 16th
    assert st(15) == "available"
    assert st(17) == "pending"                                     # newest day, still filling in
    assert st(18) == "pending"
