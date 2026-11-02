"""Phase 1: run the pipeline on your laptop with kfp local (no GCP cost).

  python run_local.py              # synthetic data, fully offline
  python run_local.py --source bq  # real GSOD via BigQuery (needs ADC: gcloud auth application-default login)

Each component runs in its own venv (SubprocessRunner) -- same isolation idea as one container per step.
Outputs land in ./local_outputs/<pipeline>-<timestamp>/<task>/...
Note: the register step is disabled locally (register=False).
"""
import argparse

from kfp import local

from pipeline import wx_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="synthetic", choices=["synthetic", "bq"])
a = ap.parse_args()

local.init(runner=local.SubprocessRunner(use_venv=True), pipeline_root="./local_outputs")
run = wx_pipeline(source=a.source, register=False)
print("pipeline finished; outputs under ./local_outputs")
