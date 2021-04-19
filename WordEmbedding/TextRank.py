import numpy as np
from collections import OrderedDict

from .PreProcessingMethods import *
from Parameters import *


class TextRank:
    def __init__(self, themePairs, stemmingOnText, deleteStopWords, kernelSize=4, dampening=0.85, steps=10, threshold=1e-5):
        self.themePairs = copy.deepcopy(themePairs)  # Copy of the global theme pair list to avoid changes to original
        self.KERNEL_SIZE = kernelSize
        self.DAMPENING = dampening
        self.STEPS = steps
        self.THRESHOLD = threshold

        if stemmingOnText:
            self.themePairs = stanfordNLPPreProcessor(self.themePairs)
        else:
            self.themePairs = splitOnSentenceAndWords(self.themePairs)

        if deleteStopWords:
            self.themePairs = removeStopWords(self.themePairs)

    def getAllKeywords(self):
        allKeywords = []
        for pair in self.themePairs:
            wordWeight = self.getKeywords(pair[0])
            wordWeight.sort(key=lambda keywordTuple: keywordTuple[0], reverse=True)
            allKeywords.append(wordWeight)
        return allKeywords

    def getKeywords(self, sentences):
        # Determine the full vocab list for the text
        vocab = self.buildVocabDict(sentences)

        # Generate the word pairs (undirected edges)
        wordPairs = self.generateWordPairs(sentences, self.KERNEL_SIZE)

        # Generate the edge matrix
        edgeMatrix = self.generateEdgeMatrix(vocab, wordPairs)

        # Init a weight matrix for each node/word (All nodes start with a weight of 1)
        nodeMatrix = np.array([1] * len(vocab))

        # Iterate through the matrices, performing the page rank equation on each step to update the node weights
        lastNodeMatrixSum = 0
        for epoch in range(self.STEPS):
            nodeMatrix = (1 - self.DAMPENING) + self.DAMPENING * np.dot(edgeMatrix, nodeMatrix)
            if abs(lastNodeMatrixSum - sum(nodeMatrix)) < self.THRESHOLD:
                break
            else:
                lastNodeMatrixSum = sum(nodeMatrix)

        # Convert dictionary into list of [score, word]
        wordWeight = []
        for word, index, in vocab.items():
            wordWeight.append([nodeMatrix[index], word])

        # Error catching for when values get too small and become NaN when transferred to numpy
        for pair in wordWeight:
            if np.any(np.isnan(pair[0])):
                pair[0] = 0

        return wordWeight

    def buildVocabDict(self, sentences):
        vocab = OrderedDict()
        index = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = index
                    index += 1
        return vocab

    def generateWordPairs(self, sentences, kernelSize):
        wordPairs = []
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + kernelSize):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in wordPairs:
                        wordPairs.append(pair)
        return wordPairs

    def generateEdgeMatrix(self, vocab, wordPairs):
        # Create empty matrix for the edge weights (vocab * vocab size)
        vocabCount = len(vocab)
        edgeMatrix = np.zeros((vocabCount, vocabCount), dtype='float')
        for word1, word2 in wordPairs:
            i, j = vocab[word1], vocab[word2]
            edgeMatrix[i][j] = 1

        # Make the matrix symmetric so the connections are undirected
        edgeMatrix = self.makeSymmetric(edgeMatrix)

        # Normalise the matrix by columns (sum of weights per node = 1)
        columnSum = np.sum(edgeMatrix, axis=0)
        edgeMatrix = np.divide(edgeMatrix, columnSum, where=(columnSum != 0))

        return edgeMatrix

    def makeSymmetric(self, matrix):
        return matrix + matrix.T - np.diag(matrix.diagonal())

    def printOrderedKeywords(self, wordWeight, number=10):
        nodeWeight = OrderedDict(sorted(wordWeight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(nodeWeight.items()):
            print(key + ' - ' + str(value))
            if i > number:
                break
