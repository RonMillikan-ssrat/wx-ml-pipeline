"""Shared library for the operational (Airflow) side of wx-ml-pipeline.

The Kubeflow pipeline in pipeline.py is the training pipeline. This package is
what Airflow runs every day: ingest observations, verify yesterday's forecast,
watch model skill, roll back a degraded model, and issue tomorrow's forecast.

Modules
  config    settings, all overridable by environment variables
  data      GSOD fetch (NCEI CSV or offline synthetic), caching, validation
  features  feature engineering, identical to the KFP make_features step
  model     train / evaluate (same model and metrics as the KFP pipeline)
  store     atomic, idempotent JSON partitions for forecasts and verifications
  registry  MLflow Model Registry: register, promote, roll back via aliases
  monitor   rolling skill of the production model from verification records
"""
