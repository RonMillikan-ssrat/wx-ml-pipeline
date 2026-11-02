#!/usr/bin/env bash
# Phase 2: one-time GCP setup for GEAP Pipelines in $WX_PROJECT (default wx-ml-pipeline-0).
# Idempotent-ish: re-running prints "already exists" errors you can ignore.
set -euo pipefail

PROJECT=${WX_PROJECT:-wx-ml-pipeline-0}
REGION=us-central1
BUCKET=gs://${PROJECT}-kfp
SA_NAME=kfp-runner
SA=${SA_NAME}@${PROJECT}.iam.gserviceaccount.com
ME=$(gcloud config get-value account)

echo "== account: $ME   project: $PROJECT"
export CLOUDSDK_CORE_PROJECT="$PROJECT"   # this script only; leaves your gcloud default alone

# 1. APIs  (GEAP still uses the aiplatform.googleapis.com service name)
gcloud services enable aiplatform.googleapis.com storage.googleapis.com \
    bigquery.googleapis.com iam.googleapis.com \
    cloudresourcemanager.googleapis.com   # aiplatform SDK looks up the project number to verify bucket ownership

# 2. Pipeline root bucket (artifacts, metrics, compiled specs land here)
gcloud storage buckets create "$BUCKET" --location="$REGION" --uniform-bucket-level-access || true

# 3. Dedicated runtime service account (least privilege vs. default compute SA)
gcloud iam service-accounts create "$SA_NAME" --display-name="KFP pipeline runner" || true

for ROLE in roles/aiplatform.user roles/bigquery.jobUser; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
      --member="serviceAccount:$SA" --role="$ROLE" --condition=None >/dev/null
done
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
    --member="serviceAccount:$SA" --role=roles/storage.objectAdmin >/dev/null

# 4. Let *you* launch jobs that run as that SA
gcloud iam service-accounts add-iam-policy-binding "$SA" \
    --member="user:$ME" --role=roles/iam.serviceAccountUser >/dev/null

# 5. Does the sklearn serving image matching scikit-learn==1.5.x in pipeline.py exist?
#    (Listing the whole vertex-ai/prediction repo takes >10 min, so check the one image.)
IMAGE=us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest
echo "== serving image: $IMAGE"
gcloud artifacts docker images describe "$IMAGE" --format="value(image_summary.digest)" 2>/dev/null || \
    echo "(not found -- pick another sklearn-cpu.* image from the 'Prebuilt containers for inference' doc page)"

echo "== done. bucket=$BUCKET  sa=$SA"
