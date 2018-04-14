#!/usr/bin/env python

import os
import csv
import numpy as np
from wordembedding import WordEmbedding
from inputbatch import InputBatch
from story import Story
from typing import List, Callable, Generator


class QAData:
    def __init__(
            self,
            maxBatchSize: int,
            embedding: WordEmbedding,
            maxBatchCount: int = None):
        self._maxBatchSize = maxBatchSize
        self._embedding = embedding
        self._maxBatchCount = maxBatchCount

    def getAnswers(
            self,
            batch: InputBatch,
            answers) -> InputBatch:
        stories = []

        for i in range(batch.size):
            split_answers = []
            last = None
            for j, tag in enumerate(answers[i]):
                if tag:
                    if last != j - 1:
                        split_answers.append([])
                    split_answers[-1].append(j)
                    last = j

            for answerIndices in split_answers:
                stories.append(Story(
                    batch.documents.ids[i], 
                    batch.documents.text[i], 
                    batch.documents.words[i], 
                    ' '.join(batch.documents.words[i][i] 
                                        for i 
                                        in answerIndices), 
                    answerIndices, 
                    '', 
                    []))

        return self._FormatBatchForModelInput(stories)

    def GenerateModelInputData(
            self,
            dataPath: str) -> Generator[InputBatch, None, None]:
        batch = []
        for stories in self._ReadDataFromCsv(dataPath):
            if len(batch) + len(stories) > self._maxBatchSize:
                if batch:
                    yield self._FormatBatchForModelInput(batch)
                    batch = []
            batch.extend(stories)
        if batch:
            yield self._FormatBatchForModelInput(batch)

    def _ReadDataFromCsv(
            self,
            path: str,
            indexDelimiter: str = ':',
            wordDelimiter: str = ' ',
            answerDelimiter: str = ',') -> List[List[Story]]:
        allStories = {}

        with open(path, encoding='utf8') as file_:
            reader = csv.reader(file_)
            next(reader, None)
            for row in reader:
                stories = allStories.setdefault(row[0], [])

                documentWords = [word.lower() 
                                    for word 
                                    in row[1].split(wordDelimiter)]

                answerIndicies = []
                for block in row[3].split(answerDelimiter):
                    start, end = (int(index) 
                                    for index 
                                    in block.split(indexDelimiter))
                    answerIndicies.extend(range(start, end))

                stories.append(Story(
                    row[0],
                    row[1],
                    documentWords,
                    ' '.join(documentWords[i] for i in answerIndicies),
                    answerIndicies,
                    row[2],
                    [word.lower() for word in row[2].split(wordDelimiter)]))

        return allStories.values()
    
    def _FormatBatchForModelInput(
            self,
            batch: List[Story]) -> InputBatch:
        batchSize = len(batch)
        
        idToIndicies = {}
        documentIds = []
        documentTexts = []
        documentWords = []
        answerTexts = []
        answerIndices = []
        questionTexts = []
        questionInputWords = []
        questionOutputWords = []

        for i, story in enumerate(batch):
            idToIndicies.setdefault(story.documentId, []).append(i)
            documentIds.append(story.documentId)
            documentTexts.append(story.documentText)
            documentWords.append(story.documentWords)
            answerTexts.append(story.answerText)
            answerIndices.append(story.answerIndices)
            questionTexts.append(story.questionText)
            questionInputWords.append(
                [self._embedding.START_WORD] + story.questionWords)
            questionOutputWords.append(
                story.questionWords + [self._embedding.END_WORD])

        maxDocumentLength = max(len(document) 
                                for document 
                                in documentWords)
        maxAnswerLength = max(len(answer) 
                                for answer 
                                in answerIndices)
        maxQuestionLength = max(len(question) 
                                for question 
                                in questionInputWords)

        documentTokens = np.zeros(
            (batchSize, maxDocumentLength), 
            dtype=np.int32)
        documentLengths = np.zeros(
            batchSize, 
            dtype=np.int32)
        answerLabels = np.zeros(
            (batchSize, maxDocumentLength), 
            dtype=np.int32)
        answerMasks = np.zeros(
            (batchSize, maxAnswerLength, maxDocumentLength), 
            dtype=np.int32)
        answerLengths = np.zeros(
            batchSize, 
            dtype=np.int32)
        questionOutputTokens = np.zeros(
            (batchSize, maxQuestionLength), 
            dtype=np.int32)
        questionLengths = np.zeros(
            batchSize, 
            dtype=np.int32)

        for i in range(batchSize):
            for j, word in enumerate(documentWords[i]):
                documentTokens[i, j] = self._embedding.Tokens.GetIndexForToken(word)
            documentLengths[i] = len(documentWords[i])

            for j, index in enumerate(answerIndices[i]):
                for k in idToIndicies[batch[i].documentId]:
                    answerLabels[k, index] = 1
                answerMasks[i, j, index] = 1
            answerLengths[i] = len(answerIndices[i])

            for j, word in enumerate(questionInputWords[i]):
                questionOutputTokens[i, j] = self._embedding.Tokens.GetIndexForToken(word)
            for j, word in enumerate(questionOutputWords[i]):
                questionOutputTokens[i, j] = self._embedding.Tokens.GetIndexForToken(word)
            questionLengths[i] = len(questionInputWords[i])

        return InputBatch(
            batchSize,
            documentIds,
            documentTexts,
            documentWords,
            documentTokens,
            documentLengths,
            answerTexts,
            answerIndices,
            answerLabels,
            answerMasks,
            answerLengths,
            questionTexts,
            np.zeros((batchSize, maxQuestionLength), dtype=np.int32),
            questionOutputTokens,
            questionLengths)