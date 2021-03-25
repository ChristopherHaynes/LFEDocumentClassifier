import math

from ..Preprocessor import *


class TermFrequency:
    def __init__(self, themePairs, removeStopWords, stemWords):
        self.themePairs = copy.deepcopy(themePairs)
        if removeStopWords:
            self.themePairs = PreProcessor.removeStopWords(self.themePairs)
        if stemWords:
            self.themePairs = PreProcessor.stemText(self.themePairs)

    def getAllTermCountsPerDocument(self):
        wordEmbeddings = []
        for pair in self.themePairs:
            wordEmbeddings.append(self.generateTermCountList(pair[0]))

    def generateAllTFIDFValues(self):
        # Separate out all the text from themes to form a corpus and track the total document count (corpus size)
        textCorpus = []
        corpusSize = 0
        for pair in self.themePairs:
            textCorpus.append(pair[0])
            corpusSize = corpusSize + 1

        # Get the TF-IDF value for each word, in each sentence, of each document in the corpus
        wordEmbeddings = []
        TESTCOUNT = 0
        for document in textCorpus:
            documentWordScores = []
            for sentence in document:
                for word in sentence:
                    tf = self.getTermFrequency(word, document)
                    idf = self.getInverseDocumentFrequency(word, textCorpus, corpusSize)
                    documentWordScores.append([tf * idf, word])
            documentWordScores.sort(key=lambda wordScoreTuple: wordScoreTuple[0], reverse=True)
            wordEmbeddings.append(documentWordScores)
            TESTCOUNT += 1
            print(TESTCOUNT)

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

    # TODO: Optimise this method, O(n^3) is far too slow. Look up table/dictionary options? REMOVE TESTCOUNT var above
    @staticmethod
    def getInverseDocumentFrequency(inputWord, corpus, corpusSize):
        # The total number of documents divided by the number of documents containing the term. Log taken of result.
        documentsContainingWord = 0
        for document in corpus:
            for sentence in document:
                if inputWord in sentence:
                    documentsContainingWord += 1
                    break

        return math.log(corpusSize / documentsContainingWord)
