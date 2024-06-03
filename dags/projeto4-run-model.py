# Execução do Modelo

# Imports
from airflow import DAG
from airflow.decorators import task
from airflow.providers.amazon.aws.operators.sagemaker_transform import SageMakerTransformOperator
from airflow.providers.amazon.aws.transfers.s3_to_redshift import S3ToRedshiftOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta

# Define variáveis de acesso
date = '{{ ds_nodash }}'                                                     # Data para o nome do job
s3_bucket = 'dsa-projeto4'                                                   # Bucket S3 usado pela instância SageMaker
test_s3_key = 'projeto4-sagemaker-xgboost-iris-prediction/test/test.csv'     # Dados de teste
output_s3_key = 'projeto4-sagemaker-xgboost-iris-prediction/output/'         # Previsões do modelo
sagemaker_model_name = "sagemaker-knn-2021-11-14-23-32-49-873"               # Nome do modelo

# Define a configuração de transformação para o SageMakerTransformOperator
transform_config = {
        "TransformJobName": "test-sagemaker-job-{0}".format(date),
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType":"S3Prefix",
                    "S3Uri": "s3://{0}/{1}".format(s3_bucket, test_s3_key)
                }
            },
            "SplitType": "Line",
            "ContentType": "text/csv",
        },
        "TransformOutput": {
            "S3OutputPath": "s3://{0}/{1}".format(s3_bucket, output_s3_key)
        },
        "TransformResources": {
            "InstanceCount": 1,
            "InstanceType": "ml.m5.large"
        },
        "ModelName": sagemaker_model_name
    }


with DAG('projeto4_model',
         start_date = datetime(2021, 10, 31),
         max_active_runs = 1,
         schedule_interval = '@daily',
         default_args = {'retries': 1, 'retry_delay': timedelta(minutes = 1), 'aws_conn_id': 'aws-sagemaker'},
         catchup = False
) as dag:

    @task
    def upload_data_to_s3(s3_bucket, test_s3_key):

        # Upload dos dados de validação para o S3. Os dados estão em: /include/dados
        s3_hook = S3Hook(aws_conn_id = 'aws-sagemaker')

        # Take string, upload to S3 using predefined method
        s3_hook.load_file(filename = 'include/dados/novos_dados.csv', key = test_s3_key, bucket_name = s3_bucket, replace = True)

    # Executa a função
    upload_data = upload_data_to_s3(s3_bucket, test_s3_key)

    # Faz as previsões
    predict = SageMakerTransformOperator(task_id = 'predict', config = transform_config)

    # Opcional - salva as previsões no RedShift
    results_to_redshift = S3ToRedshiftOperator(task_id = 'save_results',
                                               s3_bucket = s3_bucket,
                                               s3_key = output_s3_key,
                                               schema = "PUBLIC",
                                               table = "results",
                                               copy_options = ['csv'],)

    # Fluxo
    upload_data >> predict >> results_to_redshift



