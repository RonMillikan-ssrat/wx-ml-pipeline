# Task: ML on GCP with GEAP Pipelines + Kubeflow

**Goal:** Build a Kubeflow (KFP v2) pipeline that trains a next-day max-temperature model for Houston (IAH) from NOAA GSOD. Run it on your laptop first, then on **Gemini Enterprise Agent Platform (GEAP) Pipelines**, the service formerly called Vertex AI Pipelines. Finish with the model in the GEAP Model Registry.

**Project:** `wx-ml-pipeline-0` (its own GCP project) · **Region:** `us-central1` · **Account:** 98python@gmail.com

All scripts read the project from `WX_PROJECT` (default `wx-ml-pipeline-0`). For interactive `gcloud` work on this project, use the `wx-ml` configuration so your default config stays untouched:

```bash
export CLOUDSDK_ACTIVE_CONFIG_NAME=wx-ml   # this shell only
```

## How the pieces fit

| Layer | What it is here | Notes |
|---|---|---|
| Kubeflow Pipelines SDK (`kfp` 2.18) | You write the DAG in Python and compile it to IR YAML | The YAML runs the same way on kfp local, GEAP, or an OSS KFP cluster on GKE |
| GEAP Pipelines | Serverless KFP backend: no cluster to run | Every step runs as its own container on a managed VM |
| `google-cloud-aiplatform` (2.x) | SDK for submitting jobs and using the registry | The Python package name didn't change with the GEAP rename |
| BigQuery public data | `bigquery-public-data.noaa_gsod.gsod*` | Station USAF 722430 / WBAN 12960 |
| GEAP Model Registry | Versioned model plus serving container | Registered but not deployed, so nothing costs money while idle |

```
extract_gsod ─► make_features ─► train ─► evaluate ─► [register?] ─► [skill > threshold?] ─► register_model
   (BQ)          lags, split      HGB      RMSE vs                                            Model Registry
                 train <2024      sklearn  persistence
```

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Five components and the pipeline. Running it compiles `wx_tmax_pipeline.yaml` |
| `run_local.py` | Phase 1: run with `kfp local` (each step gets its own venv) |
| `setup.sh` | Phase 2: APIs, bucket, `kfp-runner` service account, IAM |
| `run_geap.py` | Phase 3: submit to GEAP Pipelines |
| `wx_tmax_pipeline.yaml` | Compiled IR. Read through it, because this file is the actual contract |

## Phase 0: Environment (Ubuntu 26.04)

```bash
python3 -m venv ~/venvs/kfp && source ~/venvs/kfp/bin/activate
pip install "kfp==2.18.0" google-cloud-aiplatform
python pipeline.py            # -> wx_tmax_pipeline.yaml
```
**Check:** open the YAML and find `root.dag.tasks`, the `executorLabel`s, the `condition-` blocks from `dsl.If`, and `resourceCpuLimit` on `train`.

## Phase 1: Local run (no cost)

```bash
python run_local.py                 # synthetic AR(1) weather, offline
python run_local.py --source bq     # real GSOD (needs: gcloud auth application-default login)
```
**Check:**
- `./local_outputs/.../evaluate/metrics` shows `rmse_F`, `rmse_persistence_F` and `skill`.
- Synthetic data gives skill ≈ 0.05, since lag-1 persistence is already a strong baseline. With real data, write down the skill you get. It determines whether the gate opens in Phase 3.
- Find where `model.joblib` was written. On GEAP, that same path becomes a `gs://` URI.

## Phase 2: GCP setup

```bash
bash setup.sh
```
**Check:**
- The script prints the available `sklearn-cpu.*` serving images. If `1-5` isn't listed, change `serving_image` and the `SKLEARN` pin in `pipeline.py` so they match. A mismatched sklearn version is the classic unpickle failure at serving time.
- Explain why the job runs as `kfp-runner` instead of the default compute SA, and what `iam.serviceAccountUser` grants your own user.

## Phase 3: Run on GEAP Pipelines

```bash
python pipeline.py && python run_geap.py
```
**Check:**
1. In the console's Pipelines view, open the run graph and each step's logs. Each step should show its own `pip install` and then the same output you saw locally.
2. Look at `gs://wx-ml-pipeline-0-kfp/pipeline_root/...` and match each artifact to its component output.
3. Re-run without changes. Every step should come back **cached** and finish in seconds. Then run `--no-cache` and compare.
4. Run `--param learning_rate=0.03 --param max_iter=600`. Only `train` and the steps after it should re-execute. Explain why, using the cache key: component spec plus inputs.
5. If the gate opened, find `wx-tmax-iah` in the Model Registry and check its `skill_x1000` label. If it didn't open, raise or lower `skill_threshold` until you see both branches.

## Phase 4: Stretch goals (pick any)

| # | Exercise | What it teaches |
|---|---|---|
| A | Swap `register_model` for GCPC `ModelUploadOp` plus a `dsl.importer` of `UnmanagedContainerModel` | Prebuilt Google components and typed artifacts |
| B | Wrap `train` in `create_custom_training_job_from_component` (GCPC) with `machine_type="n1-standard-8"` or a GPU | Running a step as a GEAP Custom Job and controlling the hardware |
| C | Add a `dsl.ParallelFor` over `learning_rate` values, then a step that picks the best model | Fan-out and fan-in with `dsl.Collected` |
| D | Batch prediction from the registered model on the 2025 test set (`Model.batch_predict`) | Serving without paying for an endpoint |
| E | `job.create_schedule(cron="0 12 * * *", ...)` for a daily retrain | GEAP pipeline schedules |
| F | Run the **same YAML** on OSS Kubeflow Pipelines (kind or GKE + kubeflow/manifests) | What managed GEAP saves you: MinIO, Argo, metadata DB |
| G | Replace sklearn with a small PyTorch MLP on a GPU step | Closer to your H100/DDP background |

## Cost and cleanup

- A pipeline run costs about $0.03 plus a few minutes of small-VM time per step. BigQuery scans for one station over 16 years are tiny.
- Endpoints bill by the hour. If you deploy one in a stretch goal, undeploy it the same day.
- Cleanup: `gcloud storage rm -r gs://wx-ml-pipeline-0-kfp/pipeline_root/**`, then delete models with `gcloud ai models list/delete --region=us-central1`. To remove everything at once: `gcloud projects delete wx-ml-pipeline-0`.

## GEAP vs OSS Kubeflow: things that differ

- GEAP requires a `pipeline_root`. OSS KFP defaults to MinIO.
- GEAP has no PVCs or recursive components. Storage is GCS, through `/gcs/` FUSE paths.
- On GEAP, caching is per-run (`enable_caching`) and per-task (`.set_caching_options`).
