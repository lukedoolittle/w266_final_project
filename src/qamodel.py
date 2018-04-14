#!/usr/bin/env python

import os
import numpy as np
import re
from tqdm import tqdm, trange
import tensorflow as tf
import logging

from tensorflow.python.layers.core import Dense # pylint: disable=E0611

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
        self._vocabSize = glove.shape[0]
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
        self._useDropout = tf.placeholder_with_default(
            False, 
            [], 
            name="useDropout")
        self._learningRate = tf.placeholder(
            tf.float32, 
            [], 
            name="learningRate")

        self._START_TOKEN = startToken
        self._END_TOKEN = endToken
        self._UNKNOWN_TOKEN = unknownToken

        self._batchLogger = logging.getLogger('batches')
        self._modelPath = modelPath
        self._modelName = modelName

    def CreateComputationGraph(
            self,
            dropoutProbability: float):
        self._dropoutKeepProb = tf.cond(
            self._useDropout,
            lambda: tf.constant(1-dropoutProbability),
            lambda: tf.constant(1.0),
            name="dropoutKeepProb")

        # construct bidirectional neural network with GRU cells
        documentEmbedding = tf.nn.embedding_lookup(
            self._MakeFancyEmbeddingLayer(
                self._embedding, 
                self._vocabSize, 
                self._dropoutKeepProb),
            self._documentTokens)
        forwardCell = tf.contrib.rnn.GRUCell(self._embeddingDimensions)
        backwardCell = tf.contrib.rnn.GRUCell(self._embeddingDimensions)
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
        answerLoss = tf.contrib.seq2seq.sequence_loss(
            logits=self._answerLogits,
            targets=self._answerLabels,
            weights=answerMask,
            name='answerLoss')

        # create the encoder
        encoderInputs = tf.matmul(
            self._encoderInputMask,
            answerOutputs,
            name='encoderInputs')
        # encoderCell = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.GRUCell(forwardCell.state_size + backwardCell.state_size)], state_is_tuple=True)
        encoderCell = tf.nn.rnn_cell.DropoutWrapper(
            tf.contrib.rnn.GRUCell(forwardCell.state_size + backwardCell.state_size),
            input_keep_prob=self._dropoutKeepProb,
            output_keep_prob=self._dropoutKeepProb)
        _, self._encoderState = tf.nn.dynamic_rnn(
            encoderCell,
            encoderInputs,
            sequence_length=self._encoderLengths,
            dtype=tf.float32,
            scope='encoder')

        # create the decoder
        decoderEmbedding = tf.nn.embedding_lookup(
            self._MakeFancyEmbeddingLayer(
                self._embedding, 
                self._vocabSize, 
                self._dropoutKeepProb),
            self._decoderInputs)
        trainingHelper = tf.contrib.seq2seq.TrainingHelper(
            decoderEmbedding,
            self._decoderLengths)
        self._projection = Dense( 
            self._embedding.shape[0],
            use_bias=False)
        # self._decoderCell = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.GRUCell(encoderCell.state_size)], state_is_tuple=True)
        self._decoderCell = tf.nn.rnn_cell.DropoutWrapper(
            tf.contrib.rnn.GRUCell(encoderCell.state_size), 
            input_keep_prob=self._dropoutKeepProb,
            output_keep_prob=self._dropoutKeepProb)
        
        decoder = tf.contrib.seq2seq.BasicDecoder(
            self._decoderCell,
            trainingHelper,
            initial_state=self._encoderState, 
            #initial_state=[self._encoderState for _ in range(1)],
            output_layer=self._projection)
        decoderOutputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            scope='decoder')
        decoderOutputs = decoderOutputs.rnn_output
        questionMask = tf.sequence_mask(
            self._decoderLengths,
            dtype=tf.float32)
        questionLoss = tf.contrib.seq2seq.sequence_loss(
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

    def _MakeFancyEmbeddingLayer(
            self,
            embedding,
            vocabularySize,
            keepProbability):
        return tf.nn.dropout(
            embedding, 
            keep_prob=keepProbability, 
            noise_shape=[vocabularySize, 1])

    # unused at current; cannot get dimensionality to match
    def _MakeFancyGRUCell(
            self,
            gruUnits,
            keepProbability,
            numberOfHiddenLayers):
        cells = []
        for _ in range(numberOfHiddenLayers):
            cell = tf.nn.rnn_cell.GRUCell(gruUnits)
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, 
                input_keep_prob=keepProbability, 
                output_keep_prob=keepProbability)
            cells.append(cell)
        return tf.nn.rnn_cell.MultiRNNCell(cells)

    def Train(
            self, 
            trainingDataGenerator, 
            numberOfEpochs: int,
            learningRate: float,
            minimumTokenCount: int = 0):
        if numberOfEpochs < 1:
            raise ValueError('Must train with 1 or more epochs')

        tf.summary.scalar(
            'loss', 
            self._loss)
        merged = tf.summary.merge_all()

        optimizer = (tf.train
                        .AdamOptimizer(learningRate)
                        .minimize(self._loss))

        saver = tf.train.Saver()
        self._session = tf.Session()
        epoch = self._loadIntermediateModel(saver)

        numberOfBatches = sum([1 for i in trainingDataGenerator()])
        epochLosses = []

        for epoch in trange(epoch + 1, numberOfEpochs + 1, desc='Epochs', unit='epoch'):
            iteration = 0
            batches = tqdm(
                trainingDataGenerator(),
                total=numberOfBatches,
                desc='Iterations',
                unit='iteration')
            for batch in batches:
                loss = 0
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
                            self._decoderLengths: batch.questions.lengths,
                            self._useDropout: True,
                            self._learningRate: learningRate
                        })
                else:
                    self._batchLogger.debug('{0} -> SKIPPED BATCH'.format(logText))
                batches.set_postfix(loss=loss)

            epochLosses.append(loss)
            trackedEpochs = len(epochLosses)
            if (trackedEpochs > 2 and 
                (epochLosses[trackedEpochs-1] > epochLosses[trackedEpochs-2]) and 
                (epochLosses[trackedEpochs-2] > epochLosses[trackedEpochs-3])):
                learningRate *= 0.5

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

    def PredictAnswers(
            self,
            batch):
        self._session = tf.Session()
        self._loadIntermediateModel(tf.train.Saver())
        
        answers = self._session.run(
            self._answerLogits, 
            {
                self._documentTokens: batch.documents.tokens,
                self._documentLengths: batch.documents.lengths,
                self._useDropout: False
            })

        return np.argmax(
            answers, 
            2)

    def PredictQuestions(
            self,
            batch,
            maximumIterations: int=16):
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self._embedding, 
            start_tokens=tf.fill(
                dims=[batch.size], 
                value=self._START_TOKEN), 
            end_token=self._END_TOKEN)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            self._decoderCell,
            helper,
            self._encoderState,
            output_layer=self._projection)
        decoderOutputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
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
                self._useDropout: False
            })
        questions[:, :, self._UNKNOWN_TOKEN] = 0
        questions = np.argmax(
            questions, 
            2)

        return questions
