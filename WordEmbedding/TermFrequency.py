import math
import copy

from .PreProcessingMethods import *


class TermFrequency:
    def __init__(self, themePairs, deleteStopWords, stemmingOnText):
        self.themePairs = copy.deepcopy(themePairs)

        if stemmingOnText:
            self.themePairs = stanfordNLPPreProcessor(self.themePairs)
        else:
            self.themePairs = splitOnSentenceAndWords(self.themePairs)

        if deleteStopWords:
            self.themePairs = removeStopWords(self.themePairs)

    def getAllTermCountsPerDocument(self):
        wordEmbeddings = []
        for pair in self.themePairs:
            wordEmbeddings.append(self.generateTermCountList(pair[0]))
        return wordEmbeddings

    def generateAllTFIDFValues(self):
        # Separate out all the text from themes to form a corpus and track the total document count (corpus size)
        textCorpus = []
        corpusSize = 0
        for pair in self.themePairs:
            textCorpus.append(pair[0])
            corpusSize = corpusSize + 1

        # Generate the DF dictionary
        dfDict = self.generateDocumentFrequencyDictionary(textCorpus)

        # Get the TF-IDF value for each word, in each sentence, of each document in the corpus
        wordEmbeddings = []
        for document in textCorpus:
            documentWordScores = []
            for sentence in document:
                for word in sentence:
                    tf = self.getTermFrequency(word, document)
                    idf = math.log(corpusSize / dfDict[word])
                    documentWordScores.append([tf * idf, word])
            documentWordScores.sort(key=lambda wordScoreTuple: wordScoreTuple[0], reverse=True)
            wordEmbeddings.append(documentWordScores)

        return wordEmbeddings

    @staticmethod
    def generateTermCountList(text):
        wordCountDict = dict()

        for sentence in text:
            for word in sentence:
                if word not in wordCountDict.keys():
                    wordCountDict[word] = 1
                else:
                    wordCountDict[word] = wordCountDict[word] + 1

        wordCountList = []
        for word, count, in wordCountDict.items():
            wordCountList.append([count, word])

        wordCountList.sort(key=lambda wordCountTuple: wordCountTuple[0], reverse=True)
        return wordCountList

    @staticmethod
    def getTermFrequency(inputWord, document):
        # Get the number of times a word appears in a given document
        totalWordCount = 0
        termFrequency = 0
        for sentence in document:
            for word in sentence:
                totalWordCount += 1
                if word == inputWord:
                    termFrequency += 1

        return termFrequency / totalWordCount

    @staticmethod
    def generateDocumentFrequencyDictionary(corpus):
        documentFrequencyDict = dict()

        # Initially collect each word as a key with value as a set of all the document indices in which that word appears
        for i in range(len(corpus)):
            for sentence in corpus[i]:
                for word in sentence:
                    try:
                        documentFrequencyDict[word].add(i)
                    except KeyError:
                        documentFrequencyDict[word] = {i}

        # Replace the set of document indices as the length of that set
        for i in documentFrequencyDict:
            documentFrequencyDict[i] = len(documentFrequencyDict[i])

        return documentFrequencyDict

    @staticmethod
    def DEPRECIATED_getInverseDocumentFrequency(inputWord, corpus, corpusSize):
        # The total number of documents divided by the number of documents containing the term. Log taken of result.
        documentsContainingWord = 0
        for document in corpus:
            for sentence in document:
                if inputWord in sentence:
                    documentsContainingWord += 1
                    break

        return math.log(corpusSize / documentsContainingWord)
