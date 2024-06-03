# Criação do Pipeline

# Imports
import io
import requests
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.providers.amazon.aws.operators.sagemaker_training import SageMakerTrainingOperator
from airflow.providers.amazon.aws.operators.sagemaker_model import SageMakerModelOperator
from airflow.providers.amazon.aws.operators.sagemaker_transform import SageMakerTransformOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta

# Variáveis de configuração
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"   # URL do dataset IRIS
date = '{{ ds_nodash }}'                                                                # Data para o nome do job
s3_bucket = 'dsa-projeto4'                                                              # Bucket S3 usado pela instância do SageMaker
input_s3_key = 'iris/processed-input-data'                                              # PATH S3 para dados de entrada
output_s3_key = 'iris/results'                                                          # PATH S3 para os resultados
role = 'your-role-arn'                                                                  # Role ARN para executar jobs SageMaker
model_name = "Iris-KNN"                                                                 # Nome do modelo
training_job_name = 'train-iris'                                                        # Nome do job de treinamento

# Definir configurações para treinamento, criação de modelo e jobs de transformação em lote
training_config = {
   "AlgorithmSpecification": {
      "TrainingImage": "405536174143.dkr.ecr.us-east-2.amazonaws.com/knn",
      "TrainingInputMode": "File"
   },
    "HyperParameters": {
      "predictor_type": "classifier",
      "feature_dim": "4",
      "k": "3",
      "sample_size": "150"
   },
   "InputDataConfig": [
      {"ChannelName": "train",
        "DataSource": {
            "S3DataSource": {
               "S3DataType": "S3Prefix",
               "S3Uri": "s3://{0}/{1}/train.csv".format(s3_bucket, input_s3_key)
            }
         },
         "ContentType": "text/csv",
         "InputMode": "File"
      }
   ],
   "OutputDataConfig": {
      "S3OutputPath": "s3://{0}/{1}/results.csv".format(s3_bucket, output_s3_key)
   },
   "ResourceConfig": {
      "InstanceCount": 1,
      "InstanceType": "ml.m5.large",
      "VolumeSizeInGB": 1
   },
   "RoleArn": role,
   "StoppingCondition": {
      "MaxRuntimeInSeconds": 6000
   },
   "TrainingJobName": training_job_name
}

model_config = {
   "ExecutionRoleArn": role,
   "ModelName": model_name,
   "PrimaryContainer": {
      "Mode": "SingleModel",
      "Image": "405536174143.dkr.ecr.us-east-2.amazonaws.com/knn",
      "ModelDataUrl": "s3://{0}/{1}/{2}/output/model.tar.gz".format(s3_bucket, output_s3_key, training_job_name),
   },
}

transform_config = {
    "TransformJobName": "test-knn-{0}".format(date),
    "TransformInput": {
        "DataSource": {
            "S3DataSource": {
                "S3DataType":"S3Prefix",
                "S3Uri": "s3://{0}/{1}/test.csv".format(s3_bucket, input_s3_key)
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
    "ModelName": model_name
    }


with DAG('projeto4_pipeline',
         start_date = datetime(2021, 10, 31),
         max_active_runs = 1,
         schedule_interval = '@daily',
         default_args = {'retries': 0, 'retry_delay': timedelta(minutes=1), 'aws_conn_id': 'aws-sagemaker'},
         catchup = False,
) as dag:

    @task
    def data_prep(data_url, s3_bucket, input_s3_key):

        # Obtém o conjunto de dados Iris, divide em dados de treinamento / teste e salva CSVs no S3
        iris_response = requests.get(data_url).content
        columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        iris = pd.read_csv(io.StringIO(iris_response.decode('utf-8')), names = columns)

        # Processando os dados
        iris['species'] = iris['species'].replace({'Iris-virginica': 0, 'Iris-versicolor': 1, 'Iris-setosa': 2})
        iris = iris[['species', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

        # Split em dados de treino e teste
        iris_train, iris_test = np.split(iris.sample(frac = 1, random_state = np.random.RandomState()), [int(0.7 * len(iris))])
        iris_test.drop(['species'], axis = 1, inplace = True)

        # Salva os arquivos no S3
        iris_train.to_csv('iris_train.csv', index = False, header = False)
        iris_test.to_csv('iris_test.csv', index = False, header = False)
        s3_hook = S3Hook(aws_conn_id = 'aws-sagemaker')
        s3_hook.load_file('iris_train.csv', '{0}/train.csv'.format(input_s3_key), bucket_name = s3_bucket, replace = True)
        s3_hook.load_file('iris_test.csv', '{0}/test.csv'.format(input_s3_key), bucket_name = s3_bucket, replace = True)

    # Prepara os dados
    data_prep = data_prep(data_url, s3_bucket, input_s3_key)

    # Treina o modelo
    train_model = SageMakerTrainingOperator(task_id = 'train_model', config = training_config, wait_for_completion = True)

    # Cria o modelo
    create_model = SageMakerModelOperator(task_id = 'create_model', config = model_config,)

    # Testa o modelo
    test_model = SageMakerTransformOperator(task_id = 'test_model', config = transform_config,)

    # Cria o fluxo
    data_prep >> train_model >> create_model >> test_model



