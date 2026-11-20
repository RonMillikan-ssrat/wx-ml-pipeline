"""MLflow Model Registry: versioning, promotion and rollback.

Every trained candidate becomes a numbered model version. Two aliases decide
what runs:
    @production  the version the daily forecast loads
    @previous    the version that was in production before the last promotion
Promotion moves @previous to the old production version and @production to
the new one. Rollback moves @production back to @previous. No model files are
copied or deleted; only the pointers move, so rollback takes seconds and
every version stays available for audit.
"""
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from . import config

# MLflow 3.18 saves sklearn models with skops, which refuses to load types it has
# not been told to trust. HistGradientBoosting needs exactly this one. It is
# safe here because every version in this registry was trained by this
# pipeline; never trust types for models from an outside source.
SKOPS_TRUSTED = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def client() -> MlflowClient:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return MlflowClient()


# Version numbers come back as int from some MLflow stores and str from others.
# Everything in this module returns str so comparisons never silently fail.
NOT_FOUND = {"INVALID_PARAMETER_VALUE", "RESOURCE_DOES_NOT_EXIST"}


def _not_found(e: MlflowException) -> bool:
    """True only for "no such alias/model". An outage is INTERNAL_ERROR and must
    propagate (so Airflow retries); treating it as "no production model" would
    make the pipeline take decisions on a wrong picture of the registry."""
    return e.error_code in NOT_FOUND


def alias_version(alias: str) -> str | None:
    try:
        return str(client().get_model_version_by_alias(config.MODEL_NAME, alias).version)
    except MlflowException as e:
        if _not_found(e):
            return None      # alias not set, or model not registered yet
        raise


def version_tags(version: str) -> dict:
    return dict(client().get_model_version(config.MODEL_NAME, version).tags)


def find_version_by_tag(key: str, value: str) -> str | None:
    """Used to make registration idempotent: a retried task finds its own earlier version."""
    try:
        versions = client().search_model_versions(
            f"name='{config.MODEL_NAME}' and tag.`{key}` = '{value}'")
    except MlflowException as e:
        if _not_found(e):
            return None
        raise
    best = max((v.version for v in versions), key=int, default=None)
    return None if best is None else str(best)


def register(model, metrics: dict, params: dict, tags: dict) -> str:
    """Log a run with the model and register it as a new version. Returns the version."""
    client()
    mlflow.set_experiment(config.EXPERIMENT)
    with mlflow.start_run(run_name=tags.get("airflow_run_id")):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        info = mlflow.sklearn.log_model(sk_model=model, name="model",
                                        registered_model_name=config.MODEL_NAME,
                                        skops_trusted_types=SKOPS_TRUSTED)
    version = str(info.registered_model_version)
    tag(version, **tags)
    return version


def tag(version: str, **tags) -> None:
    c = client()
    for k, v in tags.items():
        c.set_model_version_tag(config.MODEL_NAME, version, k, str(v))


def promote(version: str, reason: str) -> dict:
    """Make `version` production; remember the old production version as @previous."""
    c = client()
    old = alias_version(config.PROD_ALIAS)
    if old == str(version):
        return {"changed": False, "production": old, "previous": alias_version(config.PREV_ALIAS)}
    if old is not None:
        c.set_registered_model_alias(config.MODEL_NAME, config.PREV_ALIAS, old)
    c.set_registered_model_alias(config.MODEL_NAME, config.PROD_ALIAS, version)
    tag(version, status="production", promoted_reason=reason)
    if old is not None:
        tag(old, status="superseded")
    return {"changed": True, "production": str(version), "previous": old}


def rollback(from_version: str, reason: str) -> dict:
    """Move @production from `from_version` back to @previous.

    Idempotent: the caller says which version it judged bad. If production is
    no longer that version (a retry, or someone already acted), nothing happens.
    That stops a retried task from rolling back twice.
    """
    c = client()
    current = alias_version(config.PROD_ALIAS)
    if current != str(from_version):
        return {"changed": False, "production": current,
                "note": f"production is {current}, not {from_version}; nothing to do"}
    prev = alias_version(config.PREV_ALIAS)
    if prev is None:
        raise RuntimeError(f"version {from_version} is degraded but there is no @{config.PREV_ALIAS} "
                           "version to roll back to; a human has to decide")
    c.set_registered_model_alias(config.MODEL_NAME, config.PROD_ALIAS, prev)
    c.delete_registered_model_alias(config.MODEL_NAME, config.PREV_ALIAS)   # no flip-flopping
    tag(from_version, status="rolled_back", rollback_reason=reason)
    tag(prev, status="production", restored_by_rollback_of=from_version)
    return {"changed": True, "production": prev, "rolled_back": str(from_version)}


def load(alias: str = config.PROD_ALIAS):
    """Load the model behind an alias. Returns (model, version, tags)."""
    version = alias_version(alias)
    if version is None:
        raise LookupError(f"no @{alias} version of {config.MODEL_NAME} in the registry yet; "
                          "run the wx_retrain DAG first")
    client()
    model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@{alias}")
    return model, version, version_tags(version)
