"""Phase 3: submit the compiled pipeline to GEAP Pipelines (formerly Vertex AI Pipelines).

  python pipeline.py && python run_geap.py            # submit and stream status
  python run_geap.py --no-cache                       # force every step to re-run
  python run_geap.py --param max_iter=600 --param learning_rate=0.03
"""
import argparse
import os

from google.cloud import aiplatform

PROJECT = os.environ.get("WX_PROJECT", "wx-ml-pipeline-0")
REGION = "us-central1"
BUCKET = f"gs://{PROJECT}-kfp"
SA = f"kfp-runner@{PROJECT}.iam.gserviceaccount.com"

ap = argparse.ArgumentParser()
ap.add_argument("--no-cache", action="store_true")
ap.add_argument("--param", action="append", default=[], help="name=value pipeline parameter override")
ap.add_argument("--async", dest="async_", action="store_true", help="submit and return immediately")
a = ap.parse_args()


def coerce(v: str):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


params = {"project": PROJECT, "location": REGION}
params.update({k: coerce(v) for k, v in (p.split("=", 1) for p in a.param)})

aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)
job = aiplatform.PipelineJob(
    display_name="wx-tmax-iah",
    template_path="wx_tmax_pipeline.yaml",
    pipeline_root=f"{BUCKET}/pipeline_root",     # required on GEAP (no MinIO default like OSS KFP)
    parameter_values=params,
    enable_caching=not a.no_cache,
    labels={"exercise": "geap-kfp"},
)
job.submit(service_account=SA)
print("console:", job._dashboard_uri())
if not a.async_:
    job.wait()
    print("state:", job.state)
