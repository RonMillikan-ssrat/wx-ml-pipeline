# Operations runbook: Airflow + MLflow

How the forecast runs in production, and what to do when something goes wrong.
Everything below was exercised against Airflow 3.4.0 and MLflow 3.18.0 with
synthetic data; the drills section shows the observed results.

Commands assume you are in `airflow/` with the stack up. Prefix Airflow CLI
commands with `docker compose exec airflow-scheduler` (shortened below to `af`):

```bash
alias af='docker compose exec airflow-scheduler airflow'
alias wxpy='docker compose exec airflow-scheduler python'
```

## 1. What runs when

| DAG | Schedule | Catchup | What it does |
|---|---|---|---|
| `wx_retrain` | Sundays 07:00 UTC | off | Train a candidate on 2010 to (end - 90 d), score it and @production on the same 90-day holdout, register it, promote if it passes the gate |
| `wx_daily` | Daily 06:00 UTC | on, from 2026-11-01 | Wait for day D's observation, validate it, verify the forecast made for D, check rolling skill, roll back if degraded, forecast D+1 |

### Data intervals, logical dates and run ids

`wx_daily` uses `CronDataIntervalTimetable("0 6 * * *")`. Each run owns one
interval and processes the day it covers:

| Field | Example | Meaning |
|---|---|---|
| `data_interval_start` = `logical_date` | 2026-11-18 06:00 | Start of the interval; the DAG processes **D = 2026-11-18** |
| `data_interval_end` = `run_after` | 2026-11-19 06:00 | The run is created when the interval closes |
| `run_id` | `scheduled__2026-11-19T06:00:00+00:00` | Airflow 3 names runs by `run_after`, so the id shows **D+1** |

Tasks only ever use `data_interval_start.date()`, never "today". That is what
makes a run for any past date produce the same result it would have on the day.

Airflow 3 changed the default for a plain cron string to
`CronTriggerTimetable`, which has no interval (start = end = fire time). A
data-interval timetable is used here on purpose.

### Why runs go one at a time

`max_active_runs=1`. Day D's `verify` reads the forecast that day D-1's run
wrote, so runs must finish in date order. Consequence: see 4.3.

### Data source

| Item | Value |
|---|---|
| Source | NOAA GHCN-hourly, `by-year/<YYYY>/psv/GHCNh_USW00012960_<YYYY>.psv` (about 35 MB per year) |
| Daily summary | UTC day: max/min/mean temperature, mean dew point, mean sea-level pressure, mean wind, precipitation summed over routine :53 reports; days with fewer than 18 hours of temperature are treated as missing |
| Units | Converted to GSOD's (F, mb, knots, inches) so the features match the KFP pipeline |
| Lag | About 3 days behind real time (checked 2026-11-20) |
| Why not GSOD | NCEI retired GSOD on 2025-08-29; no data after that date |

Before training on real data, run the comparison with GSOD (overlap: Jan to Aug 2025):

```bash
wxpy /opt/wx/scripts/validate_ghcnh.py
```

Result on 2026-11-20 (239 days, 2025-01-01 to 2025-08-27):

| Variable | Mean diff (GHCNh - GSOD) | Mean abs diff | Correlation |
|---|---|---|---|
| temp | -0.10 F | 0.29 F | 1.000 |
| dewp | 0.11 F | 0.23 F | 1.000 |
| slp | 0.02 mb | 0.10 mb | 1.000 |
| wdsp | 0.05 kt | 0.22 kt | 0.994 |
| tmax | -2.00 F | 2.04 F | 0.987 |
| tmin | 2.06 F | 2.09 F | 0.981 |
| prcp | -0.01 in | 0.14 in | 0.489 |

Means agree almost exactly, which confirms the source and the unit
conversions. **tmax/tmin are about 2 F narrower** because they are the
extremes of the hourly readings; GSOD used the station's reported max/min,
which catch peaks between readings. The model therefore predicts "highest
hourly reading at IAH", about 2 F below the official daily maximum. Training
and serving use the same definition, so there is no skew. Precipitation
totals differ more (GSOD used different accumulation periods); it is a minor
feature.

Temperature reports per day are stable across years (2024: 36.0,
2025: 35.0, 2026: 31.8), dominated by the routine :53 report, so the daily
values are comparable between training years and the live year.

Quality codes follow the ISD convention (1/5 passed all checks, 0/4/9 passed
gross-limit checks, 2/6 suspect, 3/7 erroneous), inferred from the codes
present rather than from GHCNh documentation. Set `WX_REJECT_QC=2,3,6,7`.

## 2. First start (bootstrap)

```bash
cd airflow && cp .env.example .env
sed -i "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" .env
docker compose up -d --build
af dags unpause wx_retrain      # runs the latest completed week -> first model -> @production
                                # (first run downloads ~17 years of GHCNh, ~600 MB: several minutes)
af dags list-runs wx_retrain    # wait for success
af dags unpause wx_daily        # creates a run for every day since start_date (catchup)
```

Order matters: `forecast` fails with "no @production version" until a model exists.

`wx_retrain` has `catchup=False`, so unpausing it schedules only the most
recent completed week, not every week since `start_date`. Retraining old weeks
is pointless; reprocessing old days is not. That is the catchup decision for each DAG.

A manual run (`af dags trigger wx_retrain`) has `logical_date=None` in Airflow
3; `wx_retrain` falls back to yesterday as its end date.

`af dags test wx_daily <date>` runs one day in-process for debugging. Its
interval is inferred as the one **ending** at that date, so
`af dags test wx_daily 2026-11-12T06:00:00+00:00` processes D = 2026-11-11.

## 3. Backfills (reprocessing history)

```bash
# Always dry-run first: it prints exactly which logical dates will run.
af backfill create --dag-id wx_daily --from-date 2026-10-01 --to-date 2026-11-01 --dry-run
af backfill create --dag-id wx_daily --from-date 2026-10-01 --to-date 2026-11-01 --max-active-runs 1
```

| Option | Use |
|---|---|
| `--from-date` / `--to-date` | Compared with each run's **logical date** (06:00 on day D). `--to-date 2026-10-31` stops at D = Oct 30. To include day D, use `--to-date D+1`. |
| `--reprocess-behavior none` (default) | Only days with no run yet |
| `--reprocess-behavior failed` | Missing and failed days |
| `--reprocess-behavior completed` | Everything in the range, including successful days (e.g. after fixing a bug or changing the model) |
| `--max-active-runs 1` | Keep date order (see 1) |
| `--run-backwards` | Do not use: breaks the D-1 forecast to D verification chain |
| `--dag-run-conf '{"allow_actions": true}'` | Let a backfill roll back a model (see 5) |

What a backfill means here: forecasts for past days are **reforecasts** made
with today's @production model, not the model that was live then. A
reforecast for a date inside that model's training period is marked
`in_sample: true` and excluded from skill monitoring, because a model scored on
its own training data looks better than it is.

Outputs are keyed by date and written atomically (temp file + rename), so a
backfill overwrites rather than duplicates.

## 4. Recovery

### 4.1 Failure types and how each is handled

| Failure | Example | Handling |
|---|---|---|
| Transient | NOAA download timeout, MLflow restarting | `retries=3`, exponential backoff from 2 min (cap 30 min). Alert only after the last try fails. |
| Bad data | Impossible temperature, missing day | `AirflowFailException`: fails on the first try, no retries (retrying will not fix data). Downstream tasks become `upstream_failed`, so **no forecast is made from bad data**. Alert immediately. |
| Late data | GHCNh does not have a complete day D yet | Sensor `wait_for_obs` in reschedule mode: checks every hour, frees its slot in between, gives up after 5 days |
| Gap in the source | Day D still incomplete (< 18 hours) when a day 2+ days later is complete, e.g. IAH 2026-11-09 to 11-12 | Sensor finishes with status `missing`; `validate_obs` writes `data/gaps/date=D.json` and **skips**, so the rest of the run skips and the run still succeeds. Later days are not blocked. Forecasts whose lag inputs fall in a gap skip too, and resume after 3 complete days. |
| Dead source | File 404s, or newest complete day > 10 days old | Sensor and retrain fail immediately with "source unavailable" and alert. This is how GSOD's retirement would have shown up. |
| Registry outage | MLflow unreachable | Raises (it is never mistaken for "no model registered"), then retried as transient |

Alerts: JSON in `data/alerts/`, the task log, and `WX_ALERT_WEBHOOK` if set.

### 4.2 Re-running after a fix

```bash
# Re-run only the failed task(s) for day D and everything after them
af tasks clear wx_daily -s 2026-11-16T00:00:00 -e 2026-11-16T23:59:59 --only-failed --downstream -y

# Re-run one task (and downstream) for a range of days
af tasks clear wx_daily -t '^verify$' -s 2026-11-01 -e 2026-11-05T23:59:59 --downstream -y
```
`-s/-e` select runs by logical date. Clearing is safe because every task is
idempotent: same day in, same files out.

### 4.3 Gotcha: re-runs queue behind a waiting sensor

The newest daily run usually sits in `wait_for_obs` until GHCNh has the day, and
with `max_active_runs=1` it holds the only slot. A cleared **existing** run
(including one re-run with `--reprocess-behavior completed`) stays `queued`
until that sensor finishes. Backfills of days that have **no** run yet are not
blocked; they use the backfill's own `--max-active-runs`.
Observed: the cleared run started immediately once the sensor run completed.

## 5. Model versions, promotion and rollback

```bash
wxpy /opt/wx/scripts/registry_cli.py status        # versions, aliases, gate and rollback reasons
wxpy /opt/wx/scripts/registry_cli.py promote 7
wxpy /opt/wx/scripts/registry_cli.py rollback      # @production -> @previous
```

| Mechanism | Detail |
|---|---|
| Versioning | Every weekly candidate is a numbered MLflow model version with params, metrics and tags: Airflow run id, `train_end`, gate decision and reason |
| Aliases | `@production` is what `forecast` loads; `@previous` is the version it replaced |
| Promotion gate | Candidate skill vs persistence > 0.05 **and** holdout RMSE no worse than @production on the same days |
| Automatic rollback | `check_skill` computes rolling skill of @production over its last 30 out-of-sample verified days. With at least 14 days and skill < 0 (worse than persistence), `@production` moves to `@previous` **before** that run's forecast, so the next forecast already uses the restored model |
| Idempotent rollback | `rollback(from_version)` acts only if @production is still that version, so a retried task cannot roll back twice |
| Backfill guard | In a backfill the branch logs `WOULD ROLL BACK` and does nothing, unless the backfill was given `{"allow_actions": true}` |
| Lineage | Each forecast file records the model version, its `train_end`, the Airflow run id and the input features |

## 6. Drills (run these to see each mechanism work)

Use `WX_SOURCE=synthetic` in `.env` for drills.

| Drill | How | Observed in testing |
|---|---|---|
| Late data | Synthetic data publishes 2 days late (`WX_SYNTH_LAG_DAYS`) | Newest run: `wait_for_obs` = `up_for_reschedule`, log "not published yet" |
| Bad model, then rollback | `wxpy /opt/wx/scripts/drill_bad_model.py`, then backfill 20 days with `--reprocess-behavior completed`; then repeat the last day with `--dag-run-conf '{"allow_actions": true}'` | First pass: `WOULD ROLL BACK v3 (skill -0.525)`. With actions: `{'changed': True, 'production': '2', 'rolled_back': '3'}`; same run's forecast made by v2 |
| Registry outage | `docker compose stop mlflow`, reprocess a day, wait for `up_for_retry`, `docker compose start mlflow` | `check_skill` failed on the outage, then succeeded on the next retry; no alert sent |
| Source gap | Set `WX_DRILL_GAP_DATES=2026-11-09,2026-11-10,2026-11-11,2026-11-12`, run days Nov 8 to 15 | Every run `success`. Nov 9-12: `status: missing`, gap records written, tasks skipped. Nov 13 and 14: forecast skipped (lag inputs in the gap). Nov 15: forecast for Nov 16 issued |
| Bad data | Set `WX_DRILL_BAD_OBS_DATE=2026-11-16`, recreate containers, reprocess that day | `validate_obs` failed on its only try, downstream `upstream_failed`, alert JSON written. Removed the setting, cleared with `--only-failed --downstream`: run succeeded |

The synthetic year file is cached in `data/raw/`; delete the current year's
file after changing a drill setting so it is regenerated.
