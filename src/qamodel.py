#!/usr/bin/env python

import os
import numpy as np
import re
from tqdm import tqdm, trange
import tensorflow as tf
import logging

import tensorflow.contrib.seq2seq as seq2seq  # pylint: disable=E0611
from tensorflow.contrib.rnn import GRUCell  # pylint: disable=E0611
from tensorflow.python.layers.core import Dense  # pylint: disable=E0611


class QAModel:
    def __init__(
            self, 
            glove: np.ndarray, 
            unknownToken: int, 
            startToken: int, 
            endToken: int,
            modelPath: str,
            modelName: str = 'model'):
        self._loss = None
        self._session = None

        self._answerLogits = None
        self._decoderCell = None
        self._encoderState = None
        self._projection = None

        self._embedding = tf.get_variable(
            'embedding',
            initializer=glove)
        self._embeddingDimensions = glove.shape[1]

        self._documentTokens = tf.placeholder(
            tf.int32,
            shape=[None, None],
            name='documentTokens')
        self._documentLengths = tf.placeholder(
            tf.int32,
            shape=[None],
            name='documentLengths')
        self._answerLabels = tf.placeholder(
            tf.int32,
            shape=[None, None],
            name='answerLabels')
        self._encoderInputMask = tf.placeholder(
            tf.float32,
            shape=[None, None, None],
            name='encoderInputMask')
        self._encoderLengths = tf.placeholder(
            tf.int32,
            shape=[None],
            name='encoderLengths')
        self._decoderInputs = tf.placeholder(
            tf.int32,
            shape=[None, None],
            name='decoderInputs')
        self._decoderLabels = tf.placeholder(
            tf.int32,
            shape=[None, None],
            name='decoderLabels')
        self._decoderLengths = tf.placeholder(
            tf.int32,
            shape=[None],
            name='decoderLengths')

        self._START_TOKEN = startToken
        self._END_TOKEN = endToken
        self._UNKNOWN_TOKEN = unknownToken

        self._batchLogger = logging.getLogger('batches')
        self._modelPath = modelPath
        self._modelName = modelName

    def CreateComputationGraph(self):
        # construct bidirectional neural network with GRU cells
        documentEmbedding = tf.nn.embedding_lookup(
            self._embedding,
            self._documentTokens)
        forwardCell = GRUCell(self._embeddingDimensions)
        backwardCell = GRUCell(self._embeddingDimensions)
        answerOutputs, _ = tf.nn.bidirectional_dynamic_rnn(
            forwardCell,
            backwardCell,
            inputs=documentEmbedding, 
            sequence_length=self._documentLengths, 
            dtype=tf.float32,
            scope='answer')
        answerOutputs = tf.concat(answerOutputs, 2)
        self._answerLogits = tf.layers.dense(
            inputs=answerOutputs,
            units=2)

        # construct optimizer
        answerMask = tf.sequence_mask(
            self._documentLengths,
            dtype=tf.float32)
        answerLoss = seq2seq.sequence_loss(
            logits=self._answerLogits,
            targets=self._answerLabels,
            weights=answerMask,
            name='answerLoss')

        # create the encoder
        encoderInputs = tf.matmul(
            self._encoderInputMask,
            answerOutputs,
            name='encoderInputs')
        encoderCell = GRUCell(forwardCell.state_size + backwardCell.state_size)
        _, self._encoderState = tf.nn.dynamic_rnn(
            encoderCell,
            encoderInputs,
            sequence_length=self._encoderLengths,
            dtype=tf.float32,
            scope='encoder')

        # create the decoder
        decoderEmbedding = tf.nn.embedding_lookup(
            self._embedding,
            self._decoderInputs)
        trainingHelper = seq2seq.TrainingHelper(
            decoderEmbedding,
            self._decoderLengths)
        self._projection = Dense(
            self._embedding.shape[0],
            use_bias=False)
        self._decoderCell = GRUCell(encoderCell.state_size)
        decoder = seq2seq.BasicDecoder(
            self._decoderCell,
            trainingHelper,
            initial_state=self._encoderState, 
            output_layer=self._projection)
        decoderOutputs, _, _ = seq2seq.dynamic_decode(
            decoder,
            scope='decoder')
        decoderOutputs = decoderOutputs.rnn_output
        questionMask = tf.sequence_mask(
            self._decoderLengths,
            dtype=tf.float32)
        questionLoss = seq2seq.sequence_loss(
            logits=decoderOutputs,
            targets=self._decoderLabels,
            weights=questionMask,
            name='questionLoss')

        # create the final loss as the sum of the question and answer loss
        self._loss = tf.add(
            answerLoss,
            questionLoss,
            name='loss')
        
        return self

    def Train(
            self, 
            trainingDataGenerator, 
            numberOfEpochs: int,
            minimumTokenCount: int = None,
            maxIterations: int = None):
        if numberOfEpochs < 1:
            raise ValueError('Must train with 1 or more epochs')
            
        if self._loss is None:
            self.CreateComputationGraph()

        tf.summary.scalar(
            'loss', 
            self._loss)
        merged = tf.summary.merge_all()

        optimizer = tf.train.AdamOptimizer().minimize(self._loss)

        saver = tf.train.Saver()
        self._session = tf.Session()

        epoch = self._loadIntermediateModel(saver)

        numberOfBatches = sum([1 for i in trainingDataGenerator()])

        iteration = 0
        for epoch in trange(epoch + 1, numberOfEpochs + 1, desc='Epochs', unit='epoch'):
            batches = tqdm(
                trainingDataGenerator(),
                total=numberOfBatches,
                desc='Iterations',
                unit='iteration')
            for batch in batches:
                if maxIterations and iteration >= maxIterations:
                    break
                iteration += 1
                logText = 'iteration: {0}  token count: {1}  total document length: {2}'.format(
                    iteration,
                    len(batch.documents.tokens),
                    sum(batch.documents.lengths))
                if len(batch.documents.tokens) > minimumTokenCount:
                    self._batchLogger.debug(logText)
                    _, loss, _ = self._session.run(
                        [optimizer, self._loss, merged],
                        {
                            self._documentTokens: batch.documents.tokens,
                            self._documentLengths: batch.documents.lengths,
                            self._answerLabels: batch.answers.labels,
                            self._encoderInputMask: batch.answers.masks,
                            self._encoderLengths: batch.answers.lengths,
                            self._decoderInputs: batch.questions.inputTokens,
                            self._decoderLabels: batch.questions.outputTokens,
                            self._decoderLengths: batch.questions.lengths
                        })
                else:
                    self._batchLogger.debug('{0} -> SKIPPED BATCH'.format(logText))
                batches.set_postfix(loss=loss)

            saver.save(
                self._session,
                os.path.join(
                    self._modelPath,
                    self._modelName),
                epoch)

    def _findLargestEpoch(self):
        p = re.compile(
            r'{0}-(\d+)\.index'.format(self._modelName),
            re.IGNORECASE)
        modelFileIndicies = []
        for fileName in os.listdir(self._modelPath):
            match = p.match(fileName)
            if match:
                modelFileIndicies.append(int(match.group(1)))
        return max(modelFileIndicies) if modelFileIndicies else 0

    def _loadIntermediateModel(
            self,
            saver):
        largestEpoch = self._findLargestEpoch()
        if largestEpoch:
            saver.restore(
                self._session,
                os.path.join(
                    self._modelPath,
                    '{0}-{1}'.format(
                        self._modelName,
                        largestEpoch)))
        else:
            self._session.run(tf.global_variables_initializer())
        return largestEpoch

    def Predict(
            self, 
            testData, 
            collapse_documents, 
            expand_answers, 
            maximumIterations: int = 16):
        if self._session is None:
            self.CreateComputationGraph()
            saver = tf.train.Saver()
            self._session = tf.Session()
            self._loadIntermediateModel(saver)

        batch = collapse_documents(next(testData))

        answers = self._session.run(
            self._answerLogits, 
            {
                self._documentTokens: batch.documents.tokens,
                self._documentLengths: batch.documents.lengths,
            })
        answers = np.argmax(
            answers, 
            2)

        batch = expand_answers(
            batch, 
            answers)

        helper = seq2seq.GreedyEmbeddingHelper(
            self._embedding, 
            start_tokens=tf.fill(
                dims=[batch.size], 
                value=self._START_TOKEN), 
            end_token=self._END_TOKEN)
        decoder = seq2seq.BasicDecoder(
            self._decoderCell,
            helper,
            self._encoderState,
            output_layer=self._projection)
        decoderOutputs, _, _ = seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=maximumIterations)

        questions = self._session.run(
            decoderOutputs.rnn_output, 
            {
                self._documentTokens: batch.documents.tokens,
                self._documentLengths: batch.documents.lengths,
                self._answerLabels: batch.answers.labels,
                self._encoderInputMask: batch.answers.masks,
                self._encoderLengths: batch.answers.lengths,
            })
        questions[:, :, self._UNKNOWN_TOKEN] = 0
        questions = np.argmax(
            questions, 
            2)

        return batch, questions
