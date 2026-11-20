# MLflow tracking server and Model Registry.
FROM python:3.12-slim
RUN pip install --no-cache-dir mlflow==3.18.0
EXPOSE 5000
