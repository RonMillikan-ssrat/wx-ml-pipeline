"""Failure alerts for Airflow callbacks (no Airflow import, so it stays testable).

Every alert is written to data/alerts/ as JSON and logged. If WX_ALERT_WEBHOOK
is set (a Slack incoming webhook, for example) the message is also posted there.
"""
import json
import logging
from datetime import datetime, timezone

from . import config, store

log = logging.getLogger(__name__)


def task_failed(context) -> None:
    ti = context.get("ti") or context.get("task_instance")
    dag_run = context.get("dag_run")
    exc = context.get("exception")
    msg = {
        "time": datetime.now(timezone.utc).isoformat(),
        "dag_id": getattr(ti, "dag_id", None),
        "task_id": getattr(ti, "task_id", None),
        "run_id": getattr(dag_run, "run_id", None) or context.get("run_id"),
        "try_number": getattr(ti, "try_number", None),
        "exception": f"{type(exc).__name__}: {exc}" if exc else None,
    }
    text = (f":rotating_light: {msg['dag_id']}.{msg['task_id']} failed "
            f"(run {msg['run_id']}, try {msg['try_number']}): {msg['exception']}")
    log.error(text)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{msg['run_id']}_{msg['task_id']}")
    store.write_json(config.DATA_DIR / "alerts" / f"{safe}.json", msg)
    if config.ALERT_WEBHOOK:
        try:
            import requests
            requests.post(config.ALERT_WEBHOOK, data=json.dumps({"text": text}),
                          headers={"Content-Type": "application/json"}, timeout=10)
        except Exception as e:                       # an alert must never raise
            log.warning("alert webhook failed: %s", e)
