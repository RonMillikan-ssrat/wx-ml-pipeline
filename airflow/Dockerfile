# Airflow 3.4 plus the libraries the wx DAGs import.
# Pinning apache-airflow in the same pip install stops pip from upgrading or
# downgrading Airflow itself while it resolves the extra packages.
FROM apache/airflow:3.4.0
COPY airflow/requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir "apache-airflow==3.4.0" -r /requirements-airflow.txt
