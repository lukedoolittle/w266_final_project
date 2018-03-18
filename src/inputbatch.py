#!/usr/bin/env python

from typing import List


class QuestionCollection:
    def __init__(
            self,
            text,
            inputTokens,
            outputTokens,
            lengths):
        self.text = text
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.lengths = lengths


class AnswerCollection:
    def __init__(
            self,
            text,
            indicies,
            labels,
            masks,
            lengths):
        self.text = text
        self.indicies = indicies
        self.labels = labels
        self.masks = masks
        self.lengths = lengths


class DocumentCollection:
    def __init__(
            self,
            ids,
            text,
            words,
            tokens,
            lengths):
        self.ids = ids
        self.text = text
        self.words = words
        self.tokens = tokens
        self.lengths = lengths


class InputBatch:
    def __init__(
            self,
            batchSize: int,
            documentIds: List[str],
            documentTexts: List[str],
            documentWords,
            documentTokens,
            documentLengths,
            answerTexts: List[str],
            answerIndicies,
            answerLabels,
            answerMasks,
            answerLengths,
            questionTexts: List[str],
            questionInputTokens,
            questionOutputTokens,
            questionLengths):
        self.size = batchSize
        self.documents = DocumentCollection(
            documentIds,
            documentTexts,
            documentWords,
            documentTokens,
            documentLengths)
        self.answers = AnswerCollection(
            answerTexts,
            answerIndicies,
            answerLabels,
            answerMasks,
            answerLengths)
        self.questions = QuestionCollection(
            questionTexts,
            questionInputTokens,
            questionOutputTokens,
            questionLengths)
