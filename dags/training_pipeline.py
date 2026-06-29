from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='iris_training_pipeline',
    default_args=default_args,
    description='Pipeline to train Iris model and register to MLflow',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'iris'],
) as dag:
    
    train_model = BashOperator(
        task_id='train_model',
        bash_command='export MLFLOW_TRACKING_URI=http://mlflow:5001 && export PYTHONPATH=/opt/airflow/lab4 && cd /opt/airflow/lab4 && python train.py',
    )
