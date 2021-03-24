import math
from Parameters.AllThemes import ALL_THEMES_LIST


def generateBagOfWords(documentList, useThreshold=False, keywordPerItemThreshold=10):
    bagOfWords = set()

    for featureList in documentList:
        for i in range(len(featureList)):
            if useThreshold:
                if i < keywordPerItemThreshold:
                    bagOfWords.add(featureList[i][1])
            else:
                bagOfWords.add(featureList[i][1])

    return bagOfWords


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


def generateAllTFIDFValues(themePairs):
    # Separate out all the text from themes to form a corpus and track the total document count (corpus size)
    textCorpus = []
    corpusSize = 0
    for pair in themePairs:
        textCorpus.append(pair[0])
        corpusSize = corpusSize + 1

    # Get the TF-IDF value for each word, in each sentence, of each document in the corpus
    wordEmbeddings = []
    TESTCOUNT = 0
    for document in textCorpus:
        documentWordScores = []
        for sentence in document:
            for word in sentence:
                tf = getTermFrequency(word, document)
                idf = getInverseDocumentFrequency(word, textCorpus, corpusSize)
                documentWordScores.append([tf * idf, word])
        documentWordScores.sort(key=lambda wordScoreTuple: wordScoreTuple[0], reverse=True)
        wordEmbeddings.append(documentWordScores)
        TESTCOUNT += 1
        print(TESTCOUNT)

    return wordEmbeddings


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


def getInverseDocumentFrequency(inputWord, corpus, corpusSize):
    # The total number of documents divided by the number of documents containing the term. Log taken of result.
    documentsContainingWord = 0
    for document in corpus:
        for sentence in document:
            if inputWord in sentence:
                documentsContainingWord += 1
                break

    return math.log(corpusSize / documentsContainingWord)


def generateFeatureMask(bagOfWords, scoredText):
    featureMask = []
    scores = []
    words = []
    # Generate sub-lists for scores and text:
    for wordScoreTuple in scoredText:
        scores.append(wordScoreTuple[0])
        words.append(wordScoreTuple[1])

    for bagWord in bagOfWords:
        if bagWord in words:
            featureMask.append(scores[words.index(bagWord)])
        else:
            featureMask.append(0)
    return featureMask


def encodeThemesToValues(themes):
    targetMask = []

    for theme in themes:
        for i in range(len(ALL_THEMES_LIST)):
            if ALL_THEMES_LIST[i] == theme:
                targetMask.append(i)
                break

    # TESTING - ERROR CHECK
    if len(targetMask) != len(themes):
        print("ERROR - some themes could not be encoded")
    return targetMask


def encodePrimaryThemeToValue(themes):
    for i in range(len(ALL_THEMES_LIST)):
        if ALL_THEMES_LIST[i] == themes[0]:
            return i

