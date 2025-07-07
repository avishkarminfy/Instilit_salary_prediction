from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import os

# === DAG Default Arguments ===
default_args = {
    'owner': 'avishkar',
    'retries': 2,
    'retry_delay': timedelta(minutes=2)
}

# === Python Function to Check Drift Flag ===
def should_train():
    flag_path = "/opt/airflow/flags/drift_detected.txt"
    try:
        with open(flag_path, "r") as f:
            return f.read().strip().lower() == "yes"
    except FileNotFoundError:
        return False

# === DAG Definition ===
with DAG(
    dag_id='ml_salary_pipeline_file_triggered',
    default_args=default_args,
    description='ML pipeline triggered when new data arrives',
    schedule_interval='@hourly',  # Check every hour
    start_date=datetime(2025, 7, 1),
    catchup=False,
    tags=['ml', 'salary', 'triggered']
) as dag:

    # 1. Wait for new file to appear
    wait_for_new_data = FileSensor(
        task_id='wait_for_new_data',
        fs_conn_id='fs_default',
        filepath='data/new_data.csv',  # path relative to /opt/airflow
        poke_interval=60,  # check every 60s
        timeout=60 * 30,   # max wait: 30 mins
        mode='poke'
    )

    # 2. Detect data drift
    detect_drift = BashOperator(
        task_id='detect_drift',
        bash_command='python /opt/airflow/dags/detect_drift.py'
    )

    # 3. Check if drift detected via flag
    check_drift_flag = ShortCircuitOperator(
        task_id='check_drift_flag',
        python_callable=should_train
    )

    # 4. Retrain model if drift exists
    train_model = BashOperator(
        task_id='train_model',
        bash_command='python /opt/airflow/dags/train_model.py'
    )

    # 5. Register the model in MLflow
    register_model = BashOperator(
        task_id='register_model',
        bash_command='python /opt/airflow/dags/register_model.py'
    )

    # 6. Optional: remove data after processing to avoid retriggers
    cleanup = BashOperator(
        task_id='remove_new_data_file',
        bash_command='rm /opt/airflow/data/new_data.csv'
    )

    # DAG flow definition
    wait_for_new_data >> detect_drift >> check_drift_flag >> train_model >> register_model >> cleanup
