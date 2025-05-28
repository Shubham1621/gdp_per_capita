try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator  # Airflow 2.x
except ImportError:
    DAG = None
    PythonOperator = None
    # Airflow is not installed; this file is for Airflow DAG runner only

from datetime import datetime
import subprocess

def fetch_data():
    subprocess.run(["python", "data/fetch_data.py"], check=True)

def validate_data():
    # Placeholder for Great Expectations or custom validation
    print("Data validation step (add Great Expectations here)")

def upload_to_s3():
    # Placeholder for S3 upload logic
    print("Upload to S3 step (add boto3 logic here)")

def notify():
    print("Pipeline complete. Notify team or trigger dashboard refresh.")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1
}

if DAG is not None:
    dag = DAG(
        'fetch_gdp_data',
        default_args=default_args,
        schedule='@daily',
        catchup=False
    )
else:
    dag = None

if DAG is not None and PythonOperator is not None:
    fetch = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_data,
        dag=dag
    )

    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        dag=dag
    )

    upload = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3,
        dag=dag
    )

    notify_task = PythonOperator(
        task_id='notify',
        python_callable=notify,
        dag=dag
    )

    # Chain tasks for Airflow (suppresses 'Expression value is unused' warning)
    DAG_CHAIN = fetch >> validate >> upload >> notify_task

# Note: If you see 'Import "airflow" could not be resolved', ensure Airflow is installed in your environment:
# pip install apache-airflow
