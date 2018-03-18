#!/usr/bin/env python

from typing import List


class Story:
    def __init__(
            self,
            documentId: str,
            documentText: str,
            documentWords: List[str],
            answerText: str,
            answerIndices: List[int],
            questionText: str,
            questionWords: List[str]):
        self.documentId = documentId
        self.documentText = documentText
        self.documentWords = documentWords
        self.answerText = answerText
        self.answerIndices = answerIndices
        self.questionText = questionText
        self.questionWords = questionWords
