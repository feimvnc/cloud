"""
Sequence2Sequence Algorithm
Take a string of inputs, produces another string of output

Regression algorithm: produce a number
Classification algorithm: predict one among N classes

seq2seq use cases:
Neural machine translation (Improvement BLUE scores quite a bit)
Summarization (2k words to one paragraph)
Speech to text

seq2seq
word to word
phrase to phrase
sentence to sentence
document level translation

Core of this is: RNN/LSTM

RNN 
https://colah.github.io/posts/2015-08-Understanding-LSTMs/

    Ht          # hidden state
    ^
 -- |  --
|-> A  -|       # neuron, N
    ^
    |
    Xt

Neuron network can suffer vanishing gradient or eploding gradient.

Long Short Term Memory (LSTM)

Encoder Decoder Architecture (new architecture 2014)

                            -> target lang 1
source language -> encoding -> target lang 2
                            -> target lang 3

There may not have enough data to train all language pairs.

greedy algorithm, always choose maximum weight
beam search, similar to breadth-first-search in a tree

Attention mechanism

attention: which set of words subset of the neural network focus on to translate

The BLEU score seems to drop after 15 words, and reached the lowerest when there are 60-70 words.

NMT attention (neural machine translation)

Attention is all you need, you don't even need LSTM, this is the transformer architecture.

BLEU (BiLingual Evaluation Understudy)

"The cat caught the mouse."  (candidate string)

Translated to below, but which one was good?

"The cat caught the rat."  (reference string)
"The mouse was caught by the cat."
"The kitty caught the mouse."

Score 1 (complete match) - 0 (not match)
One gram, two gram, three gram, four gram scores

-Quick and inexpensive to score
-Easy to understand
-Language independent
-Intuitive
-Wide adoption

Mxnet Sockeye: A toolkit for Neural Machine Translation

Amazon seq2seq is based on Sockeye

"""

# s3 bucket and prefix
bucket = 'sagemaker-pr-useast1'
prefix = 'sagemaker/demo-seq2seq'

import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()

# import libraries
from time import gmtime, strftime
import time
import numpy as np
import os
import json

# For plotting attention matrix
import matplotlib
%matplotlib inline 
import matplotlib.pyplot as plt 

# Download datasets, Conference on Machine Translation (WMT) 2017
%%bash
wget http://data.statmt.org/smt17/translation-task/preprocessed/de-en/corpus.tc.de.gz & \
wget http://data.statmt.org/smt17/translation-task/preprocessed/de-en/corpus.tc.en.gz & wait
gunzip corpus.tc.de.gz &
gunzip corpus.tc.en.gz & wait
mkdir validation
curl http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz | tar xvzf - -C validation 

# It is a common practices to split words into subwords using Byte Pair Encoding (BPE).

# train only first 10000 lines
!head -n 10000 corpus.tc.en > corpus.tc.en.small
!head -n 10000 corpus.tc.de > corpus.tc.de.small

#Create vocabulary mapping (strings to integer)
#Convert these files to x-recordio-protobuf as required for training by SageMaker Seq2seq

%%bash 
#python3 create_vocab_proto.py

# Preprocessing
%%time
%%bash
python3 create_vocab_proto.py \
    --train-source corpus.tc.en \
    --train-target corpus.tc.de \ 
    --val-source validation/newstest2014.tc.en \
    --val-target validation/newstest2014.tc.de 

# upload pre-processed datasets and vocab to s3
def upload_to_s3(bucket, prefix, channel, file):
    s3 = boto3.resource('s3')
    data = open(file, 'rb')
    key = prefix + "/" + channel + "/" + file
    s3.bucket(bucket).put_object(Key=key, Body=data)

region_name = boto3.Session().region_name 

from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(region_name, 'seq2seq')
print('Using SageMaker seq2seq container: {} ({})'.format(container, region_name))

# Training the Machine Learning model
job_name = 'Demo-seq2seq-en-db-' + strftime('%Y-%m-%t-%H', gmtime())
print('Training job', job_name)

create_training_params = \
{
    "AlgorithmSpecification": {
        "TrainingImage": container,
        "TrainingInputMode": "File"
    },
    "RoleArn": role,
    "OutputDataConfig": {
        "S3OutputPath": "s3://{}//{}".format(bucket, prefix)
    },
    "ResourceConfig": {
        #Seq2seq dose not support multiple machines. Currently it only support single machine, multiple GUPs
        "InstanceCount": 1,
        "InstanceType": "ml.p3.16xlarge", # suggest one of ["ml.p2.16xlarge", "ml.p2.8xlarge", "ml.p2.xlarge"]
        "VolumeSizeInGB": 50
    },
    "TrainingJobName": job_name,
    "HyperParameters": {
        #Please refer to the documentation for complete list of parameters
        "max_seq_len_source": "60",
        "max_seq_len_target": "60",
        "optimized_metric": "bleu",
        "batch_size": "64", # Please use a larger batch size (256, 512) if using ml.p2.8xlarge or ml.p2.16xlarge
        "checkpoint_frequency_num_batches": "1000",
        "run_num_hidden": "512",
        "run_num_hidden": "512",
        "num_layers_encoder": "1",
        "num_layers_decoder": "1",
        "num_embed_source": "512",
        "num_embed_target": "512",
        "checkpoint_threshold": "3",
        "max_num_batches": "2100"
        # Training will stop after 2100 iterations/batches
        # Remove above if you want to better model.
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 48 * 3600
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3Datatype": "S3Prefix",
                    "S3Uri": "s3://{}//{}/train/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
        },
        {
            "ChannelName": "vocab",
            "DataSource": {
                "DataSource": "S3Prefix",
                "S3Uri": "s3://{}//{}/vocab/".format(bucket, prefix),
                "S3DataDistributionType": "FullReplicated"
            },
        },
        {
            "ChannelName":"validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/validation/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
        }
    ]
}

sagemaker_client = boto3.Session().client(service_name="sagemaker")
sagemaker_client.create_training_job(**create_training_params)

status = sagemaker_client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print(status)

status = sagemaker_client.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print(status)
#If job failed, determine why
if status == 'Failed':
    message = sagemaker_client.describe_training_job(TrainingJobName=job_name)['FailureReason']
    print('Training failed with the following error: {}').format(message)
    raise Exception("Training job failed")

# Inference
# Translate English to German
# Create model - create a model using the artifact (model.tar.gz) produced by training
# Create Endpoint Configuration - create configuration defining an endpoint using the above model
# Create Endpoint - Use the configuration to create an inference endpoint
# Perform inference - Perform inference on some input data using the endpoint

# Create model
use_pretrained_model = False 

# Use a pretrained model
# If you want as training might take hours/days to complete
# Uncomment below to use pre-trained model
#use_pretrained_model = True
#model_name = "demo-pretrained-en-de-model"
#!curl https://s3-us-west-2.amazonaws.com/seq2seq-data/model.tar.gz > model.tar.gz
#!curl https://s3-us-west-2.amazonaws.com/seq2seq-data/vocab.src.json > vocab.src.json
#!curl https://s3-us-west-2.amazonaws.com/seq2seq-data/vocab.trg.json > vocab.trg.json
#upload to s3(bucket, prefix, 'pretrained_model', 'model.tar.gz')
#model_data = "s3://{}//{}/pretrained_model/model.tar.gz".format(bucket, prefix)

%%time
sage = boto3.client('sagemaker')

if not use_pretrained_model:
    info = sage.describe_training_job(TrainingJobName=job_name)
    model_name=job_name
    model_data=info['ModelArtifacts']['S3ModelArtifacts']

print(model_name)
print(model_data)

primary_container = {
    'Image': container,
    'ModelDataUri': model_data
}

create_model_response = sage.create_model(
    ModelName=model_name,
    ExecutingRoleArn=role,
    PrimaryContainer=primary_container
)
print(create_model_response['ModelArn'])

# Create endpoint configuration
from time import gmtime, strftime

endpoint_config_name = 'demo-Seq2SeqEndpointConfig-' + strftime('%Y-%m-%d-%H-%M-%S', gmtime())
print(endpoint_config_name)
create_endpoing_config_response = sage.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType": "ml.m4.xlarge',
        'InitialInstanceCount': 1,
        'ModelName': model_name,
        'VariantName': 'AllTraffic'
    }])

print('Endpoint config Arn:' + create_endpoing_config_response['EndpointConfigArn'])

#Create endpoint
%%time
import time

endpoint_name = 'demo-Seq2SeqEndpoint-' + strftime("%Y-%m-%d-%H-%M-%S", gtmtime())
print(endpoint_name)
create_endpoint_response = sage.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)
print(create_endpoint_response['EndpointArn'])

#You now have a functioning infernce endpoint
runtime = boto3.client(service_name='runtime_sagemaker')

resp = sage.describe_endpoing(runtime)
status = resp['EndpointStatus']
print("Status: "+ status)

sage.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)

endpoint_response = sage.describe_endpoint(EndpointName=endpoint_name)
status = endpoint_response['EndpointStatus']
print('Endpoint creation ended with EndpointStatus = {}').format(status)

if status != 'InService':
    raise Exception('Endpoint creaton failed.')

# Perform Inference
# Using JSON format for inference (Suggested for a single or small number of data instances)
sentences = ['You are so good !',
             "can you drive a car ?",
             "I want to watch a movie ."]
payload = {"instances": []}
for sent in sentences:
    payload["instances"].append({"data": sent})

response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                   ContentType="application/json",
                                   Body=json.dumps(payload))

response = response['Body'].read().decode('utf-8')
response = json.loads(response)
print(response)

# Retrieve the attention matrix
sentence = 'you are so good'
payload = {"instances": [{
                'data': sentence,
                'configuration': {'attetion_metric':'true'}
                }
            ]}

response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                   ContentType='application/json',
                                   Body=json.dumps(payload))

response = response['Body'].read().decode('utf-8')
response = json.loads(response)['predictions'][0]

source = sentence
target = response['target']
print('Summary %s \nTarget: %s' % (source, target))

# Define a function for plotting the attention matrics
def plot_matrix(attention_matrix, target, source):
    source_tokens = source.split()
    target_tokens = target.split()
    assert attention_matrix.shape[0] == len(target_tokens)
    plt.imshow(attention_matrix.transpose(), interpolation='nearest', cmap='Greys')
    plt.xlabel('target')
    plt.ylabel('source')
    plt.gca().set_xticks([i for i in range(0, len(target_tokens))])
    plt.gca().set_yticks([i for i in range(0, len(target_tokens))])

    plt.gca().set_xticklabels(target_tokens)
    plt.gca().set_yticklabels(source_tokens)
    plt.tight_layout()

# Use prodobuf format for inference (suggested for efficient bulk inference)
# Reading the vocabulary mappings as this mode of inference accepts list of integers and returns list of integers
import io
import tempfile
from record_pb2 import Record
from create_vocab_proto import vocab_from_json, reverse_vocab, write_recordio, list_to_record_bytes, read_next

source = vocab_from_json('vocab.src.json')
target = vocab_from_json('vocab.trg.json')

source_rev = reverse_vocab(source)
target_rev = reverse_vocab(target)

sentences = ['this is so cool',
             'i am having dinner .',
             'i am sitting in an aeroplane .',
             'come let us go for a long drive .']

#Convert strings to integers using source vocab mapping.
#Out-of-vocabulary strings are mapped to 1 - the mapping for < 
sentences = [[source.get(token, 1) for token in sentence.split()] for sentence in sentences]
f = io.BytesIO()
for sentence in sentences:
    record = list_to_record_bytes(sentence, [])
    write_recordio(f, record)

response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType='application/x-recordio-protobuf',
                                   Body=f.getvalue())
response = response['Body'].read()

#Now parse the protobuf response and covnert list of integers back to strings
def _parse_proto_response(received_bytes):
    output_file = tempfile.NamedTemporaryFile()
    output_file.write(received_bytes)
    output_file.flush()
    target_sentences = []
    with open(output_file.name, 'rb') as datum:
        next_record = True
        while next_record:
            next_record = read_next(datum)
            if next_record:
                rec = Record()
                rec.ParseFromString(next_record)
                target = list(rec.features["target"].int32_tensor.values)
                target_sentences.append(target)
            else:
                break
    return target_sentences

targets = _parse_proto_response(response)
resp = [" ".join([target_rev.get(token, "<unk>") for token in sentence]) for sentence in targets]
print(resp)

# Stop/Close the endpoint
sage.delete_endpoint(EndpointName=endpoint_name)
