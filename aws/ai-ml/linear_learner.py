"""
"The organized thinker never gives up anything he undertakes until he has exhausted every effort to finish it." - Andrew Carnegie
"All models are wrong, but some are useful." - George Box

The SageMaker built-in Linear Learner can train as a binary, or multi-classification model, as well as linear regression.

The Binary Classifier is a great tool for asking Yes or No questions.
For multi-classification rather than 0 or 1, we hav 0 to (n-1) classes.
Liner Regression - Linear means we are going to make a prediction based on a linear function of the input features.

linear = sagemaker estimator Estimator(params)
linear set_hyperparameters(predictor_type="binary_classifier")

Linear Learner is not a deep learning, it is a distributed stochastic gradient descent.


"""
# borrow code from others
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
import boto3
import re 
from sagemaker import get_execution_role
import os 
import io
import time 
import json 
import sagemaker.amazon.common as smac

role = get_execution_role()  # capture whoiam

bucket = 'linear-learner-sagemaker'
prefix = 'housing'  # subfolder

#Data points
mypoints = array([[1,1.2],[1,3],[1.5,4.1],[2,5],[3,6],[3.5,2],[3.75,4],[4.6,5],[5,8],[6,6],[8,7.5]])
#plot a scatter chart
plt.scatter(mypoints[:,0], mypoints[:,1])
plt.annotate("",
             xy=(1,3), xycoords='data',
             xytext=(8,8), textcoords='data',
             arrowprops-dict(arrowstyle="-",
                             connectionstyle="arc3,rad=0."),)
plt.show()

# Linear Learner accepts CSV or recordIO-wrapped protobuf
# CSV = No headers and label in first column
# recordIO-wrapped = best practice = Pipe Mode
# Check Correlation and Standard Deviation

#Download data using Boston housing price dataset
s3 = boto3.resource("s3")
KEY = prefix + '/housing_boston.csv'  # file name
s3.Bucket(bucket).download_file(KEY, 'housing_boston.csv')  # download to notebook instance

# Intro do pandas
data = pd.read_csv('housing_boston.csv')
print(data.shape)

# Manual task, canned datasets provide the column names and meanings
# Note: target column should always be first 
data.columns = ['MDEV','CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

display(data.head())
display(data.describe())
data.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'))  # corelation


# Training
# Linear Learner = a distributed implementation of stochastic gradient descent
# Train multiple models simutaneously L1 (Ordinary Least Squares), L2(Ridge Regression - add a constraint to the coefficient)
# Various optimizer Adam Adagrad SGD
# Before training starts, you should understand what your success criteria is
"""
The linear learner supports FIVE metrics
Objective Loss - mean value of the loss function
    Binary classification = Logistic loss
    Linear Regression = Squared loss
Binary Classification accuracy
Binary F beta (or F1)
Precision
Recall

Accuracy = Number of correct True Positives and True Negatives from all observations

Precision = of all our predicted Positive observations what percent are actual positive
-Precision is a good metric to train against when the costs of False Positive is high

Recall = of all actual positives what percent did we get correct
-Recall is a good metric to train against when the cost of False Negative is high

F1 = harmonic balance between Precision and Recall
-Train to F1 when you need a balance between False Negative and False Positive
"""

# Intro to numpy 
rand_split = np.random.rand(len(data))

# split dataset into 80% training, 15% validation, 5% testing
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.95)
test_list = rand_split >= -.95

data_train = data[train_list]
data_val = data[val_list]
data_test = data[test_list]

print(data_train)

# create label to data
# Return value True if >22.0, or 0 otherwise
train_y = ((data_train.iloc[:,0] > 22.0)+0).as_matrix()
# Value of all rows, column 1 to rest
train_x = data_train.iloc[:,1:].as_matrix()

print('{}, {}'.format(train_y.shape, train_x.shape))
#(403,),(403,13)   # 403 y, 13 features

val_y = ((data_val.iloc[:,0]>22.0)+0).as_matrix()
val_x = data_val.iloc[:1:].as_matrix()

test_y = ((data_test.iloc[:,0] > 22.0)+0).as_matrix()
text_x = data_test.iloc[:,1:].as_matrix()

# Declare a filename for training data in s3
train_file = 'linera_train.data'

# Convert to RecordIO-wrapped protobuf
f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, train_x.astype('float32'), train_y.astype('float32'))
f.seek(0) # reset memory back

# Upload to s3
s3.bucket(bucket).Object(prefix+'/train/'+train_file).upload_fileobj(f)

# Get a reference to an LinearLearner container image
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'linear-learner')

linear_job = "DeepDive-linear-" + time.strftime("%Y-%m-%y-%H-%M-%S", time.gmtime())
print("Job name is:", linear_job)

linear_training_params = {
    "RoleArn": role,
    "TrainingJobName": linear_job,
    "AlgorithmSpecification": {
        "TrainingImage", container,
        "TrainingInputMode": "File"  # uses File mode, not Pipe
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.c4.2xlarge",  # donot need GPU
        "VolumnSizeInGB": 10
    },
    "InputDataConfig": [  # for training and validation
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Url": "s3://{}//{}/train/".format(bucket, prefix),
                    "S3DataDistributionType": "SharedByS3Key"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Url": "s3://{}//{}/validation/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        },
    ],
    "OutputDataConfig": {
        "S3OutputPath": "s3://{}/{}".format(bucket, prefix)
    },
    "HyperParameters": {
        "feature_dim": "14",
        "mini_batch_size": "100",
        "predictor_type": "regressor",
        "epoch": "10",
        "num_models": "32",  # can train up to 32 versions of models
        "loss": "absolute_loss"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 60 * 60
    }
}

# Train
%%time

sm = boto3.client('sagemaker')
sm.create_training_job(**linear_training_params)
status = sm.describe_training_job(TrainingJobName=linear_job)['TrainingJobStatus']
print(status)
sm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=linear_job)  # wait till complete or failure
if status == 'Failed':
    message = sm.describe_training_job(TrainingJobName=linear_job)['FailureReason']
    print(message)
    raise Exception('Training job failed')

# Prepare a model context for hosting to run inference
linear_hosting_container = {
    'Image': container,
    'ModelDataUrl': sm.describe_training_job(TrainingJobName=linear_job)['ModelArtifacts']['S3ModelArtifacts']
}

create_model_response = sm.create_model(
    ModelName=linear_job,
    ExecutionRoleArn=role,
    PrimaryContainer=linear_hosting_container)

print(create_model_response['ModelArn'])

# finally create an endpoint configuration (!= endpoint)
linear_endpoint_config = "Deepdive-linear-endpoint-config-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print(linear_endpoint_config)

create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName=linear_endpoint_config,
    ProductionVariants=[{
        "InstanceType": "ml.m4.xlarge",
        "InitialInstanceCount": 1,
        "ModelName": linear_job,
        "VariantName": "AllTraffic"
    }])

print("Endpoing configuration: " + create_endpoint_config_response['EndpointConfiguration'])

# deploy the endpoint to SageMaker hosting
linear_endpoint = 'Deepdive-linear-endpoint' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print(linear_endpoint)

create_endpoint_response = sm.create_endpoint(
    EndpointName=linear_endpoint,
    EndpointConfigName=linear_endpoint_config
)
print(create_endpoint_response['EndpointArn'])

resp = sm.describe_endpoint(EndpointName=linear_endpoint)
status = resp['EndpointStatus']
print('Status: ' + status)

sm.get_waiter('endpoint_in_service').wait(EndpointName=linear_endpoint)

resp = sm.describe_endpoint(EndpointName=linear_endpoint)
status = resp["EndpointStatus"]
print("Arn: " + resp['EndpointArn'])
print("Status: " + status)

if status != 'InService':
    raise Exception('Endpoint creation did not succeed')

# Helper function to convert an array to CSV
def np2csv(arr):
    csv = io.BytesIO()
    np.savetxt(csv, arr, delimiter=',', fmt='%g')
    f = open("test_observations.csv", "w")
    f.write(csv.getvalue().decode())
    return csv.getvalue().decode().rstrip()

# Lets send our test file (as a csv file) to the endpoint to get prediction
# get endpoint name on console or above
runtime = boto3.client('runtime.sagemaker')
payload = np2csv(test_x)
response = runtime.invoke_endpoint(EndpointName='DeepDive-linear-endpoint-2025123456789',
                                   ContentType="text/csv",
                                   Body=payload)

result = json.loads(response['Body'].read().decode())
test_pred = np.array([r['score'] for r in result['predictions']])

print(test_pred)
print(result)

# Classification threshold of 0.5
test_pred_class = (test_pred > 0.5)+0
test_pred_baseline = np.repeat(np.median(train_y), len(test_y))

prediction_accuracy = np.mean((test_y == test_pred_class))*100
baseline_accuracy = np.mean((test_y == test_pred_baseline))*100

print("Prediction accuracy: ", round(prediction_accuracy, 1), "%")
print("Baseline accuracy: ", round(baseline_accuracy,1), '%')
