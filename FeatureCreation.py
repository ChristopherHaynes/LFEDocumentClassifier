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

