""" 
When to choose Random Cut Forest?
When the application requires greater accuracy and repeatability is not paramount.
When lower variance is more desired than lower bias.

Bias - assumptions made that attempt to make f(x) easier to learn 
Variance - the amount of change in f(x) from different datasets

"""

import boto3
import botocore
import sys
import pandas as pd 
import urllib.request 
from sagemaker import RandomCutForest
import matplotlib 
import matplotlib.pyplot as plt

s3 = boto3.resource('s3')
bucket = 'demo-rcf'
prefix = 'rcf-deepdive'
execution_role = sagemaker.get_execution_role()

f = 'nyc_taxi.csv'
data_source = 'https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv'

urllib.request.urlretrieve(data_source, f)
taxi_data = pd.read_csv(f, delimiter=',')

taxi_data.head()

%matplotlib inline 
matplotlib.rcParams['figure.dpi'] = 100
taxi_data.plot()

#notebook magic command help
#magic

session = sagemaker.Session()

# training job spec
rcf = RandomCutForest(role=execution_role,
                      train_instance_count=1,
                      train_instance_type='ml.x4.xlarge',
                      data_location='s3://{}/{}/'.format(bucket, prefix),
                      output_path='s3://{}/{}/output'.format(bucket, prefix),
                      num_samples_per_tree=512,
                      num_trees=50)

# rcf.fit(rcf.record_set(taxi_data.value.as_matrix().reshape(-1,1)))


newData = taxi_data.value.as_matrix()
print(type(newData))
print(newData.shape)  # numpy.ndarray

newerData = newData.reshape(-1, 1)
print(newerData.shape)
print(type(newerData))  # numpy.ndarray

type(taxi_data)

newerData = rcf.record_set(newerData)
print(newerData)

rcf.fit(twitter_data)

rcf_inference = rcf.deploy(
    initial_instance_count = 1,
    instance_type="ml.m4.xlarge"
)




