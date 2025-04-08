"""
Time series forecasting with linear learner
-Using Linear Regression to forecast monthly demand

Design principals, How we choose which algorithm to build.

We strongly favor record i/o protobuf for file format
e.g. for image file, this can reduce file size for easy and quick read
Also, it is simple way to partition and simplify a distributed setting

Time Series Forecasting
Use cases:
Supply chain efficienties by doing demand prediction avoid outages
Allocating computing resources more effectively by predicting webserver traffic
Saving lives by forecasting patient visits and staffing hospitals to meet patient needs

In warehouse time series forecasting is used to predict product and labor demand in fullfillment centers, e.g. prime day, black friday, cyber monday

It is a mix of objective statistics and subjective interpretations 

What's the right acclivity daily weekly monthly


Cold start forecasting: with little of no  historical data 

Stationary Series

A stocastic process whose unconditional joint probability distribution does not change when shifted in time. Consequently, parameters such as mean and variance also do not change over time.

"""

## Example 1

bucket = 'demo-air'
prefix = 'sagemaker/demo-linear-time-series-forecast'

# Define iam role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()

# import python libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import time
import json
import sagemaker.amazon.common as smac
import sagemaker
from sagemaker.predictor import csv_serializer, json_deserializer

# download gasoline dataset
!wget http://robjhyndman.com/data/gasoline.csv_serializer

# look at data  
gas = pd.read_csv('gasonline.csv', header=None, names=['thousands_barrels'])  # volumes
display(gas.head())
plt.rcParams['figure.figsize'] = [20,5]
plt.plot(gas)

# Break the dataset to easily ingest
gas['thousands_barrels_lag1'] = gas['thousands_barrels'].shift(1)
gas['thousands_barrels_lag2'] = gas['thousands_barrels'].shift(2)
gas['thousands_barrels_lag3'] = gas['thousands_barrels'].shift(3)
gas['thousands_barrels_lag4'] = gas['thousands_barrels'].shift(4)

gas = gas.iloc[4:, ]
split_train = int(len(gas) * 0.6)
split_test = int(len(gas) * 0.8)

train_y = gas['thousands_barrels'][:split_train]
train_X = gas.drop('thousands_barrels', axis=1).iloc[:split_train, ].as_matrix()
validation_y = gas['thousands_barrels'][split_train:split_test]
validation_X = gas.drop('thousands_barrels', axis=1).iloc[split_test:, ].as_matrix()
test_y = gas['thousands_barrels'][split_test:]
test_X = gas.drop('thousands_barrels', axis=1).iloc[split_test:, ].as_matrix()

# Convert to recordIO-wrapped protubuf, upload to s3, and start training data
buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, np.array(train_X).astype('float32'), np.array(train_y).astype('float32'))
buf.seek(0)

key = 'linear_train.data'
boto3.resoruce('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))

key = 'linear_validation.data'
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation', key)).upload_fileobj(buf)
s3_validation_data = 's3://{}/{}/validation/{}'.format(bucket, prefix, key)
print('uploaded validation data location: {}'.format(s3_validation_data))

# Train
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'linear-leaner')

# Kick off training job using SageMaker distributed managed training
sess = sagemaker.Session()
linear = sagemaker.estimator.Estimator(container,
                                       role,
                                       training_instance_count=1
                                       train_instance_type='ml.c4.xlarge',
                                       output_path='s3://{}/{}/output'.format(bucket, prefix),
                                       sagemaker_session=sess)
linear.set_hyperparameters(feature_dim=59,
                           min_batch_size=100,
                           predictor_type='regressor',
                           epochs=10,
                           num_modules=32,
                           loss='absolute_loss')

linear.fit({'train': s3_train_data, 'validation': s3_validation_data})

# Host model
linear_predictor = linear.deploy(initial_instance_count=1,
                                instance_type='ml.m4.xlarge')

# Forecast
# There are metrics to measure forecast error
# Root mean square error (rmse)
# Mean absolute percent error (mape)
# Geometric mean of the relative absolute error (GMRAE)
# Quantile forecast errors
# Errors that account for asymmetric loss in over or under prediction
# Median absolute percent error (MdAPE)

# Generate naive forecast
gas['thousands_barrels_lag52'] = gas['thousands_barrels'].shift(52)
gas['thousands_barrels_lag104'] = gas['thousands_barrels'].shift(104)
gas['thousands_barrels_naive_forecast'] = gas['thousands_barrels_lag52'] ** 2 / gas['thousands_barrels_lag104']
naive = gas[split_test:]['thousands_barrels_naive_forecast'].as_matrix()

# Investigating accuracy
print('Naive MdAPE = ', np.median(np.abs(test_y - naive) / test_y))
plt.plot(np.array(test_y), label='actual')
plt.plot(naive, label='naive')
plt.legend()
plt.show()

# Generate one step ahead forecast
linear_predictor.content_type = 'text/csv'
linear_predictor.serializer = csv_serializer
linear_predictor.deserializer = json_deserializer

# Invoke endpoint for prediction
result = linear_predictor.predict(test_X)
one_step = np.array([r['score'] for r in result['predictions']])

# compare forecast errors
print('One-step-ahead MdAPE = ', np.median(np.abs(test_y - one_step) / test_y))
plt.plot(np.array(test_y), label='actual')
plt.plot(one_step, label='forecast')
plt.legend()
plt.show()

# Requirements
"""
Must be a stationary dataset
If not, stationarize with detrending, differencing, and seasonality

Gzipped JSON line, or parquet file

DeepAR parameter
likelihood: Gaussian, beta, student tau, deterministic

"""


## Example 2
# SageMaker/DeepAR demo on electricity dataset
# Steps:
# Prepare the dataset
# Use the SageMaker python sdk to train a deepAR model and deploy it
# Make requests to the deployed model to obtain forecast interactively
# Illustrate advanced features of DeepAR: missing values, additional time features, non-regular frequencies, and category information

%matplotlib inline 

import sys
from urllib.request import urlretrieve 
import zipfile 
from dateutil.parser import parse
import json
from random import shuffle
import random 
import datetime
import os

import boto3
import s3fs 
import sagemaker 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgts as widgets
from ipywidgets import IntSlider, FloatSlider, Checkbox

# Set random seeds for reproducibility
np.random.seed(271828)
random.seed(271828)

sagemaker_session = sagemaker.Session()

# s3 bucket should be in the same region as notebook instance
s3_bucket = sagemaker.Session().default_bucket() # replace
s3_prefix = 'deepar-electricity-demo-notebook'

role = sagemaker.get_execution_role()

region = sagemaker_session.boto_region_name

s3_data_path = "s3://{}/{}/data".format(s3_bucket, s3_prefix)
s3_output_path  ="s3://{}/{}/output".format(s3_bucket, s3_prefix)

# Configure container image
image_name = sagemaker.amazon.amazon_estimator.get_image_uri(region, 'forecasting-deepar', 'latest')

# Import electricity dataset and upload to s3
DATA_HOST = 'https://archieve.ics.uci.edu'
DATA_PATH = '/ml/machine-learning-databases/00321'
ARCHIVE_NAME = 'LD2011_2014.txt.zip'
FILE_NAME = ARCHIVE_NAME[:-4]

def progress_report_hook(count, block_size, total_size):
    mb = int(count * block_size // 1e6)
    if count % 500 == 0:
       sys.stdout.write('\r{} MB downloaded'.format(mb))
       sys.stdout.flush()

if not os.path.isfile(FILE_NAME):
    print("downloading dataset (258MB), can take a few minutes depending on your connection")
    urlretrieve(DATA_HOST + DATA_PATH + ARCHIVE_NAME, ARCHIVE_NAME, reporthook=progress_report_hook)

    print("\nextracting data archive")
    zip_ref = zipfile.ZipFile(ARCHIVE_NAME, 'r')
    zip_ref.extractall("./")
    zip_ref.close()
else:
    print("File found skipping download")

# Load, parse, convert to pandas collection time series

data = pd.read_csv(FILE_NAME, sep=";", index_col=0, parse_dates=True, decimal=',')
num_timeseries = data.shape[1]
data_kw = data.resample('2H').sum() / 8  # resample with granularity of 2 hours
timeseries = []
for i in range(num_timeseries):
    timeseries.append(np.trim_zeors(data_kw.iloc[:,i], trim='f'))

# Let's plot data
fig, axs = plt.subplots(5, 2, figsize=(20, 20), sharex=True)
axx = axs.ravel()
for i in range(0, 10):
    timeseries[i].loc['2014-01-01':'2014-01-14'].plot(ax=axx[i])
    axx[i].set_xlabel('date')
    axx[i].set_ylabel('KW consumption')
    axx[i].grid(which='minor', axis='x')

# Train and test split
# Use 2 hour frequency for the time series
freq = '2H'

# We predict for 7 days
prediction_length = 7 * 12    

# We use 7 days as context length
context_length = 7 * 12

# Specify portion of data for training, from to dates
start_dataset = pd.Timestamp("2014-01-01 00:00:00", freq=freq)
end_training = pd.Timestamp("2014-09-01 00:00:00")

# DeepAR json input format represents each time series as a json object, contain start/target value
training_data = [
    {
        "start": str(start_dataset),
        "target": ts[start_dataset:end_training-1].tolist()
    } for ts in timeseries
]
print(len(training_data))

# Test data, extending beyond the training range, used for computing test scores, 
# Forecast the trailing 7 days predictions
num_test_windows = 4
test_data = [
    {
        "start": str(start_dataset),
        "target": ts[start_dataset:end_training + k * prediction_length].tolist()
    } 
    for k in range(1, num_test_windows + 1)
    for ts in timeseries
]
print(len(test_data))

# Write jsonlines file format that DeepAR understands, also support gzipped jsonlines and parquet
def write_dicts_to_file(path, data):
    with open(path, 'wb') as fp:
        for d in data:
            fp.write(json.dumps(d).encode('utf-8'))
            fp.write("\n".encode('utf-8'))

%%time
write_dicts_to_file("train.json", training_data)
write_dicts_to_file("test.json", test_data)

# Copy to s3
s3 = boto3.resource('s3')
def copy_to_s3(local_file, s3_path, override=False):
    assert s3_path.startswith('s3://')
    split = s3_path.split('/')
    bucket = split[2]
    path = '/'.join(split[3:])
    buk = s3.Bucket(bucket)

    if len(list(buk.objects.filter(Prefix=path))) > 0:
        if not override:
            print('File s3://{}/{} already exists.\nSet override to upload anyway.\n'.format(s3_bucket, s3_path))
            return 
        else:
            print('Overwriting existing file')
    with open(local_file, 'rb') as data:
        print('Uploading file to {}'.format(s3_path))
        buk.put_object(Key=path, Body=data)

%%time
copy_to_s3("train.json", s3_data_path + '/train/train.json')
copy_to_s3("test.json", s3_data_path + "/test/test.json")        

# Train a model
estimator = sagemaker.estimator.Estimator(
    sagemaker_session=sagemaker_session,
    image_name=image_name,
    role=role,
    train_instance_count=1,
    train_instance_type='m1.c4.2xlarge',
    base_job_name='deepar-electricity-demo',
    output_path=s3_output_path
)

# Set hyperparameters
hyperparameters = {
    "time_freq": freq,
    "epochs": "400",
    "early_stopping_patience": "40",
    "mini_batch_size": "64",
    "learning_rate": "5E-4",
    "context_length": str(context_length),
    "prediction_length": str(prediction_length)
}

estimator.set_hyperparameter(**hyperparameters)

# Launch training job
%%time
data_channels = {
    "train": "{}/train/".format(s3_data_path),
    "test": "{}/test/".format(s3_data_path)
}
estimator.fit(inputs=data_channels, wait=True)

# Create endpoint and predictor

# Utility class using pandas.Series objects rather than raw JSON strings
class DeepARPredictor(sagemaker.predictor.RealTimePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, content_type=sagemaker.content_types.CONTENT_TYPE_JSON, **kwargs)

    def predict(self, ts, cat=None, dynamic_feat=None, num_samples=100, return_samples=False, quantiles=["0.1","0.5", "0.9"]):
        """
        Predict time series listed in 'ts'

        ts - pandas.Series object, the time series to predict
        cat - integer, the group associated to the time series (default: None)
        num_samples - boolean indicating whether to include samples in the response (default: False)
        quantiles - list of strings specifying the quantiles to compute (default: ["0.1", "0.5", "0.9"])

        Return value: list of pandas.DataFrame objects, each containing the predictions

        """
        prediction_time = ts.index[-1] + 1
        quantiles = [str(q) for q in quantiles]
        req = self.__encode_request(ts, cat, dynamic_feat, num_samples, return_samples, quantiles)
        res = super(DeepARPredictor, self).predict(req)
        return self.__decode_response(res, ts.index.freq, prediction_time, return_samples)

    def __encode_request(self, ts, cat, dynamic_feat, num_samples, return_samples, quantiles):
        instance = series_to_dict(ts, cat if cat is not None else None, dynamic_feat if dynamic_feat else None)

        configuration = {
            "num_samples": num_samples,
            "output_types":['quantiles', 'samples'] if return_samples else ['quantiles'],
            'quantiles': quantiles
        }
        #TODO
        http_request_data = {

        }

        #TODO
        #return [ x if np.Series(x) else Noen for x in data]

    def series_to_dict(ts, cat=None, dynamic_feat=None):
        obj = {"start": str(ts.index[0]), 
                "target": encode_target(ts)}
        if cat is not None:
            obj["cat"] = cat
        if dynamic_feat is not None:
            obj["dynamic_feat"] = dynamic_feat
        return obj

predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge',
    predictor_cls=DeepARPredictor
)

# Make predictions
predictor.predict(ts=timeseries[120], quantiles=[0.10, 0.5, 0.90])

# define plot function
def plot(
    predictor,
    target_ts,
    cat=None,
    dynamic_feat=None,
    forecast_date=end_training,
    show_samples=False, 
    plot_history=7 * 12,
    confidence=80
):
    print("calling served model to generate predictions starting from {}".format(str(forecast_date)))
    assert(confidence > 50 and confidence < 100)
    low_quantile = 0.5 - confidence * 0.005
    up_quantile = confidence * 0.005 + 0.5

    # we first construct the argument to call our model
    args = {
        "ts": target_ts[:forecast_date],
        "return_samples": show_samples,
        "quantiles": [low_quantile, 0.5, up_quantile],
        "num_sample": 100
    }

    if dynamic_feat is not None:
        args["dynamic_feat"] = dynamic_feat
        fig = plt.figure(figsize=(20, 6))
        ax = plt.subplot(2,1,1)
    else: 
        fig = plt.figure(figsize=(20, 3))
        ax = plt.subplot(1,1,1)

    if cat is not None:
        args['cat'] = cat
        ax.text(0.9, 0.9, 'cat = {}'.format(cat), transform=ax.transAxes)

    prediction = predictor.predict(**args)

    # plot samples
    if show_samples:
        for key in prediction.keys():
            if "sample" in key:
                prediction[key].plot(color='')


style = {'description_width': 'initial'}

@interact_manual(
    customer_id=IntSlider(min=0,max=369,value=91,style=style),
    forecast_day=IntSlider(min=0,max=100,value=51,style=style),
    confidence=IntSlider(min=60,max=95,value=80,step=5,style=style),
    show_samples=Checkbox(value=False),
    continuous_update=False
)

def plot_interact(customer_id, forecast_day, confidence, history_weeks_plot, show_samples):
    plot(
        predictor,
        target_ts=timeseries[customer_id],
        forecast_date=end_training + datetime.timedelta(days=forecast_day),
        show_samples=show_samples,
        plot_history=history_weeks_plot * 12 * 7,
        confidence=confidence
    )
    
# Delete endpoint
predictor.delete_endpoint()
#predictor_new_feature.delete_endpoint()






