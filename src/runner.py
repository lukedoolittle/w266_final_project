#!/usr/bin/env python

import argparse
import os
import itertools

import logging

from qamodel import QAModel
from wordembedding import WordEmbedding
from qadata import QAData

parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode',
    help='operation mode (either train or predict)',
    #required=True,
    type=str,
    default='train')
parser.add_argument(
    '--epochs',
    help='number of training epochs',
    type=int,
    default=5)
parser.add_argument(
    '--maxbatchsize',
    help='max size of each batch',
    type=int,
    default=32)
parser.add_argument(
    '--minimumtokencount',
     help='minimum number of tokens in a batch',
     type=int,
     default=15)  #0 in paper, this is a hack
parser.add_argument(
    '--embeddingdimension',
     help='local path to embedding file',
     type=int,
     default=300)
parser.add_argument(
    '--trainingfile',
     help='local path to training file',
     type=str,
     default='train.csv')
parser.add_argument(
    '--testingfile',
     help='local path to testing file',
     type=str,
     default='test.csv')
parser.add_argument(
    '--dropoutprobability',
     help='rate of dropout in model',
     type=float,
     default=.3)
parser.add_argument(
    '--learningrate',
     help='learning rate for the optimizer',
     type=float,
     default=.0002)
args = parser.parse_args()

embeddings_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    '../data/',
    'glove.6B.{0}d.txt'.format(args.embeddingdimension))
training_data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    '../data/',
    args.trainingfile)
testing_data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    '../data/',
    args.testingfile)
models_directory = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'models')

if not os.path.exists(models_directory):
    os.makedirs(models_directory)

# create the logger
logger = logging.getLogger('batches')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 
    'batches.log'))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

print('Loading pretrained word embeddings')
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

print('Creating computational graph')
model.CreateComputationGraph(
    args.dropoutprobability)

if args.mode == 'train':
    print('Preparing for training')
    model.Train(
        lambda: data.GenerateModelInputData(training_data_path),
        args.epochs,
        args.learningrate,
        args.minimumtokencount)
elif args.mode == 'predict':
    # its possible we need to remove duplicate documents here
    batch = next(data.GenerateModelInputData(testing_data_path))
    answers = model.PredictAnswers(batch)
    questions = model.PredictQuestions(data.getAnswers(
                                        batch, 
                                        answers))

    for i in range(len(questions)):
        question = itertools.takewhile(
            lambda t: t != embeddings.END_TOKEN,
            questions[i])
        print('Q: ' + ' '.join(embeddings.Tokens.GetTokenForIndex(token) 
                                        for token 
                                        in question))
        print('A: ' + batch.answers.text[i])
else:
    raise ValueError('Invalid mode specified: {0}'.format(args.mode))