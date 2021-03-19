
def generateBagOfWords(documentList):
    bagOfWords = set()

    for featureList in documentList:
        for featureTuple in featureList:
            bagOfWords.add(featureTuple[1])

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
