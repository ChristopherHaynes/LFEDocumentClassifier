import numpy as np
import nltk as nltk
import spacy
import string
from collections import OrderedDict

nlp = spacy.load('en_core_web_sm')


class TextRank:
    def __init__(self, themePairs, kernelSize=4, dampening=0.85, steps=20, threshold=1e-5):
        self.themePairs = themePairs.copy()  # Copy of the global theme pair list to avoid changes to original
        self.KERNEL_SIZE = kernelSize
        self.DAMPENING = dampening
        self.STEPS = steps
        self.THRESHOLD = threshold

        self.splitOnSentenceAndWords()
        self.removeStopWords()
        self.stemText()

    def getKeywords(self, sentences):
        # Determine the full vocab list for the text
        vocab = self.buildVocabDict(sentences)

        # Generate the word pairs (undirected edges)
        wordPairs = self.generateWordPairs(sentences, self.KERNEL_SIZE)

        # Generate the edge matrix
        edgeMatrix = self.generateEdgeMatrix(vocab, wordPairs)

        # Init a weight matrix for each node/word (All nodes start with a weight of 1)
        nodeMatrix = np.array([1] * len(vocab))

        # Iterate through the matrices, performing the pagerank equation on each step to update the node weights
        lastNodeMatrixSum = 0
        for epoch in range(self.STEPS):
            nodeMatrix = (1 - self.DAMPENING) + self.DAMPENING * np.dot(edgeMatrix, nodeMatrix)
            if abs(lastNodeMatrixSum - sum(nodeMatrix)) < self.THRESHOLD:
                break
            else:
                lastNodeMatrixSum = sum(nodeMatrix)

        # Create a sorted list from highest ranked word to lowest
        wordWeight = []
        for word, index, in vocab.items():
            wordWeight.append([nodeMatrix[index], word])

        return wordWeight

    def splitOnSentenceAndWords(self):
        for i in range(len(self.themePairs)):
            doc = nlp(self.themePairs[i][0])

            sentences = []
            for sent in doc.sents:
                selectedWords = []
                for token in sent:
                    if str(token) not in string.punctuation:
                        selectedWords.append(token)
                sentences.append(selectedWords)
            self.themePairs[i][0] = sentences

    def stemText(self):
        stemmer = nltk.stem.PorterStemmer()

        for i in range(len(self.themePairs)):
            newText = []
            for sentence in self.themePairs[i][0]:
                newSentence = []
                for word in sentence:
                    newSentence.append(stemmer.stem(str(word)))
                newText.append(newSentence)
            self.themePairs[i][0] = newText

    def removeStopWords(self):
        stopWords = set(nltk.corpus.stopwords.words('english'))

        for i in range(len(self.themePairs)):
            filteredText = []
            for sentence in self.themePairs[i][0]:
                newSentence = []
                for word in sentence:
                    if str(word) not in stopWords:
                        newSentence.append(str(word))
                filteredText.append(newSentence)
            self.themePairs[i][0] = filteredText

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
