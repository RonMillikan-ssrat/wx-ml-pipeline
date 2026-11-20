"""wx_daily: one run per observation day D.

    wait_for_obs ─► validate_obs ─► verify ─► check_skill ─► skill_branch ─┬─► rollback ──┐
    (sensor)        (data QC)       (score    (rolling      (decide)      └─► skill_ok ──┴─► forecast
                                    forecast  skill of                                    (Tmax for D+1,
                                    for D)    @production)                                 trigger rule
                                                                                           none_failed_min_one_success)

Scheduling
  CronDataIntervalTimetable("0 6 * * *") gives each run a data interval
  [D 06:00, D+1 06:00) UTC. The run for D is created when that interval
  closes, at D+1 06:00. Every task works on D = data_interval_start.date() and
  never on "today", which is what makes catchup and backfills correct.
  (Airflow 3's default for a cron string is CronTriggerTimetable, which has
  no interval: the run fires at the cron time and data_interval_start ==
  data_interval_end. A data-interval timetable is the right model for
  "process the day that just finished".)

Ordering
  max_active_runs=1: the verify step for D reads the forecast that D-1's run
  wrote, so runs must go in date order. Do not backfill with --run-backwards.

Recovery
  - Network and other transient errors: retries=3 with exponential backoff.
  - Bad data (DataValidationError): AirflowFailException fails immediately,
    without retries, because retrying will not fix bad data.
  - Late data: the sensor waits in reschedule mode (it frees its worker slot
    between checks) for up to 5 days, then fails and alerts.
  - Dead source: a missing file (404) or a feed more than 10 days behind
    fails the sensor immediately. This is how GSOD's retirement in 2025
    would have shown up: as an alert, not as a sensor waiting forever.
  - Every task failure calls wxml.alerts.task_failed.
  - All outputs are written atomically and keyed by date, so clearing a task
    or re-running a day overwrites rather than duplicates.

Rollback
  If the rolling skill of @production over the last 30 out-of-sample verified
  days falls below WX_ROLLBACK_SKILL (default 0, i.e. worse than persistence)
  with at least WX_ROLLBACK_MIN_N days of evidence, @production moves back to
  @previous before tomorrow's forecast is made. Backfill runs only report what
  they would do, unless triggered with conf {"allow_actions": true}.
"""
import functools
from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, get_current_context, task
from airflow.sdk.bases.sensor import PokeReturnValue
from airflow.sdk.exceptions import AirflowFailException
from airflow.timetables.interval import CronDataIntervalTimetable

from wxml import alerts, config, data, ops, registry


def obs_day():
    ctx = get_current_context()
    start = ctx.get("data_interval_start")
    if start is None:
        raise AirflowFailException("this run has no data interval; trigger it with a logical date")
    return start.date()


def hard_fail_on_bad_data(fn):
    """Turn DataValidationError into AirflowFailException (fail now, no retries)."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except data.DataValidationError as e:
            raise AirflowFailException(f"data validation failed: {e}") from e
        except data.SourceUnavailable as e:
            raise AirflowFailException(f"source unavailable: {e}") from e
    return wrapper


@dag(
    dag_id="wx_daily",
    schedule=CronDataIntervalTimetable("0 6 * * *", timezone="UTC"),
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    catchup=True,
    max_active_runs=1,
    default_args={
        "retries": 3,
        "retry_delay": timedelta(minutes=2),
        "retry_exponential_backoff": True,
        "max_retry_delay": timedelta(minutes=30),
        "on_failure_callback": alerts.task_failed,
    },
    tags=["wx", "operations"],
    doc_md=__doc__,
)
def wx_daily():

    @task.sensor(poke_interval=timedelta(hours=1), timeout=timedelta(days=5), mode="reschedule")
    def wait_for_obs() -> PokeReturnValue:
        """Wait until GHCNh has a complete day D (it lags about 3 days)."""
        d = obs_day()
        try:
            ready = data.observation_available(d)
        except data.SourceUnavailable as e:         # missing or stalled feed: waiting won't help
            raise AirflowFailException(f"source unavailable: {e}") from e
        except Exception as e:                      # a network blip is "not yet", not a failure
            print(f"availability check failed, will try again: {e}")
            ready = False
        print(f"observation for {d}: {'available' if ready else 'not published yet'}")
        return PokeReturnValue(is_done=ready)

    @task
    @hard_fail_on_bad_data
    def validate_obs() -> dict:
        return ops.validate_day(obs_day())

    @task
    @hard_fail_on_bad_data
    def verify() -> dict:
        rec = ops.verify_day(obs_day())
        print(rec)
        return rec

    @task
    def check_skill() -> dict:
        stats = ops.check_skill(obs_day())
        print(stats)
        return stats

    @task.branch
    def skill_branch(stats: dict) -> str:
        if not stats["degraded"]:
            return "skill_ok"
        ctx = get_current_context()
        conf = ctx["dag_run"].conf or {}
        if ctx["dag_run"].run_type == "backfill" and not conf.get("allow_actions"):
            print(f"WOULD ROLL BACK v{stats['model_version']} (skill {stats['skill']:.3f}), "
                  "but this is a backfill; pass conf {'allow_actions': true} to act")
            return "skill_ok"
        return "rollback"

    @task
    def rollback(stats: dict) -> dict:
        result = registry.rollback(
            stats["model_version"],
            reason=f"rolling skill {stats['skill']:.3f} < {config.ROLLBACK_SKILL} "
                   f"over {stats['n']} days ending {stats['window_end']}")
        print(result)
        return result

    @task
    def skill_ok() -> None:
        pass

    @task(trigger_rule="none_failed_min_one_success")
    @hard_fail_on_bad_data
    def forecast() -> dict:
        rec = ops.forecast_day(obs_day(), get_current_context()["run_id"])
        print({k: v for k, v in rec.items() if k != "features"})
        return rec

    stats = check_skill()
    choice = skill_branch(stats)
    wait_for_obs() >> validate_obs() >> verify() >> stats
    choice >> [rollback(stats), skill_ok()] >> forecast()


wx_daily()
