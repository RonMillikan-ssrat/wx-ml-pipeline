"""wx_retrain: weekly champion/challenger retraining.

    refresh_history ─► train_candidate ─► register ─► promotion_gate ─┬─► promote
                                                                      └─► reject

  train_candidate  trains on 2010 .. (end - 90 days), then scores the candidate
                   AND the current @production model on the same 90-day holdout.
  register         every candidate becomes a new MLflow model version (tagged
                   with this Airflow run id, so a retry reuses it instead of
                   registering a duplicate).
  promotion_gate   promote only if candidate skill vs persistence > WX_SKILL_GATE
                   and its holdout RMSE is no worse than production's.
  promote          @previous <- old production, @production <- candidate.

catchup=False: retraining for every missed week would be pointless; only the
latest data matters. Compare wx_daily, where every missed day matters.
"""
from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, get_current_context, task
from airflow.sdk.exceptions import AirflowFailException
from airflow.timetables.interval import CronDataIntervalTimetable

from wxml import alerts, config, data, ops, registry


def end_day():
    """Last full day covered by this run (fallback: yesterday, for manual runs)."""
    end = get_current_context().get("data_interval_end")
    end = end or datetime.now(timezone.utc)
    return end.date() - timedelta(days=1)


@dag(
    dag_id="wx_retrain",
    schedule=CronDataIntervalTimetable("0 7 * * 0", timezone="UTC"),   # Sundays 07:00 UTC
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "retry_exponential_backoff": True,
        "on_failure_callback": alerts.task_failed,
    },
    tags=["wx", "training"],
    doc_md=__doc__,
)
def wx_retrain():

    @task
    def refresh_history() -> dict:
        """Download/summarize every year once (about 35 MB each); refresh the current year."""
        end = end_day()
        try:
            data.fetch_year(end.year, force=True)              # newest data for the current year
            df = data.load_range(datetime(config.TRAIN_START_YEAR, 1, 1).date(), end)
        except data.SourceUnavailable as e:
            raise AirflowFailException(f"source unavailable: {e}") from e
        complete = df[df.tmax.notna()]
        return {"days": int(len(df)), "complete_days": int(len(complete)),
                "first": str(df.date.min()), "last_complete": str(complete.date.max())}

    @task
    def train_candidate(_history: dict) -> dict:
        result = ops.train_candidate(end_day(), get_current_context()["run_id"])
        print({k: v for k, v in result.items() if k != "model_path"})
        return result

    @task
    def register(result: dict) -> dict:
        version = ops.register_candidate(result, get_current_context()["run_id"])
        print(f"registered {config.MODEL_NAME} version {version}")
        return {**result, "version": version}

    @task.branch
    def promotion_gate(result: dict) -> str:
        ok, why = ops.gate(result)
        print(("PROMOTE: " if ok else "REJECT: ") + why)
        registry.tag(result["version"], gate_decision="promote" if ok else "reject", gate_reason=why)
        return "promote" if ok else "reject"

    @task
    def promote(result: dict) -> dict:
        out = registry.promote(result["version"], reason="passed weekly gate")
        print(out)
        return out

    @task
    def reject(result: dict) -> None:
        registry.tag(result["version"], status="rejected")

    reg = register(train_candidate(refresh_history()))
    promotion_gate(reg) >> [promote(reg), reject(reg)]


wx_retrain()
