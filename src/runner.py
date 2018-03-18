#!/usr/bin/env python

import argparse
import os
import numpy as np
from tqdm import tqdm, trange
import itertools

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq   # pylint: disable=E0611
from tensorflow.contrib.rnn import GRUCell   # pylint: disable=E0611
from tensorflow.python.layers.core import Dense   # pylint: disable=E0611

import logging

from qamodel import QAModel
from wordembedding import WordEmbedding
from qadata import QAData

parser = argparse.ArgumentParser()
parser.add_argument(
    '--epochs',
    help='number of training epochs',
    type=int, default=5)
parser.add_argument(
    '--maxbatchsize',
    help='max size of each batch',
    type=int, default=64)
parser.add_argument(
    '--maxiterations',
    help='maximum number of batches per epoch',
    type=int, default=None)
parser.add_argument(
    '--minimumtokencount',
     help='minimum number of tokens in a batch',
     type=int,
     default=15)
args = parser.parse_args()

embeddings_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    '../data', 
    'glove.6B.100d.txt')
training_data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    '../data/', 
    'train.csv')
testing_data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    '../data/', 
    'test.csv')
models_directory = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'models')

if not os.path.exists(models_directory):
    os.makedirs(models_directory)

logger = logging.getLogger('batches')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 
    'batches.log'))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

embeddings = WordEmbedding(
    '<UNK>',
    '<START>',
    '<END>').LoadEmbeddings(embeddings_path)

data = QAData(
    args.maxbatchsize,
    embeddings)

model = QAModel(
    embeddings.Glove,
    embeddings.UNKNOWN_TOKEN,
    embeddings.START_TOKEN,
    embeddings.END_TOKEN,
    models_directory)
model.CreateComputationGraph()

model.Train(
    lambda: data.GenerateModelInputData(training_data_path),
    args.epochs,
    args.minimumtokencount)

# batch, questions = model.Predict(data.GenerateModelInputData(testing_data_path), None, None)
# for i in range(batch.size):
#     question = itertools.takewhile(
#         lambda t: t != embeddings.END_TOKEN,
#         questions[i])
#     print('Question: ' + ' '.join(look_up_token(token) for token in question))
#     print('Answer: ' + batch.answers.text[i])
#     print()