#!/usr/bin/env python

import os
import io
import numpy as np
from typing import Optional
import logging

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector # pylint: disable=E0611
from tqdm import tqdm

class TokenDictionary:
    def __init__(self):
        self._tokenDictionary = {}
        self._tokenReverseDictionary = []
        self._default = None

    def AddDefault(
            self,
            word: str):
        self._default = self.AddToken(word)
        return self._default

    def AddToken(
            self,
            word: str):
        index = len(self._tokenReverseDictionary)
        self._tokenDictionary[word] = index
        self._tokenReverseDictionary.append(word)
        return index

    def GetIndexForToken(
            self,
            word: str):
        return self._tokenDictionary.get(
            word, 
            self._default)

    def GetTokenForIndex(
            self,
            token: int):
        return self._tokenReverseDictionary[token]

    def GetCount(self):
        return len(self._tokenReverseDictionary)


class WordEmbedding:
    def __init__(
            self,
            unknownWord,
            startWord,
            endWord):
        self.Tokens = TokenDictionary()

        self.UNKNOWN_WORD = unknownWord
        self.START_WORD = startWord
        self.END_WORD = endWord

        self.UNKNOWN_TOKEN = self.Tokens.AddDefault(self.UNKNOWN_WORD)
        self.START_TOKEN = self.Tokens.AddToken(self.START_WORD)
        self.END_TOKEN = self.Tokens.AddToken(self.END_WORD)

        self.Glove = None

    def LoadEmbeddings(
            self,
            embeddingFilePath: str):
        with open(embeddingFilePath, encoding='utf8') as fileHandle:
            dimensions = len(fileHandle.readline().split(' ')) - 1
            fileHandle.seek(0)
            vocabularySize = sum(1 for line in fileHandle) + 3
            fileHandle.seek(0)

            self.Glove = np.ndarray(
                (vocabularySize, dimensions), 
                dtype=np.float32) # pylint: disable=E1101
            self.Glove[self.UNKNOWN_TOKEN] = np.zeros(dimensions)
            self.Glove[self.START_TOKEN] = -np.ones(dimensions)
            self.Glove[self.END_TOKEN] = np.ones(dimensions)

            for line in fileHandle:
                chunks = line.split(' ')
                index = self.Tokens.AddToken(chunks[0])
                self.Glove[index] = [float(chunk) for chunk in chunks[1:]]
                if self.Tokens.GetCount() >= vocabularySize:
                    break
        
        return self