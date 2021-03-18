import pandas as pd
from summa import keywords
from rake_nltk import Rake
from collections import OrderedDict

from PreProcessor import PreProcessor
from TextRank import TextRank
from BagOfWords import BagOfWords
from NeuralNet import NeuralNet

# Mode 1 for RAKE, Mode 2 for bespoke implementation
MODE_TYPE = 1


def printOrderedKeywords(wordWeight, number=10):
    nodeWeight = OrderedDict(sorted(wordWeight.items(), key=lambda t: t[1], reverse=True))
    for i, (key, value) in enumerate(nodeWeight.items()):
        print(key + ' - ' + str(value))
        if i > number:
            break


print("Reading File...")
dataFile = pd.read_excel("C:\\Users\\Chris\\Desktop\\Data\\lfeData.xlsx", engine='openpyxl')
print("File Loaded.")
pp = PreProcessor(dataFile)

if MODE_TYPE == 1:
    pp.extractThemePairs(MODE_TYPE)

    r = Rake()
    for i in range(len(pp.themePairs)):
        r.extract_keywords_from_text(pp.themePairs[i][0])
        pp.themePairs[i][0] = r.get_ranked_phrases_with_scores()

if MODE_TYPE == 2:
    pp.extractThemePairs(MODE_TYPE)

    print("Theme Pairs Length: " + str(len(pp.themePairs)))
    print(pp.themePairs[3][0])

    tr = TextRank()
    themePairKeywords = []
    print("Using TextRank to extract keywords.")
    for i in range(len(pp.themePairs)):
        wordWeight = tr.getKeywords(pp.themePairs[i][0])
        themePairKeywords.append(wordWeight)

    keywordsList = [item for sublist in themePairKeywords for item in sublist]
    keywordsSet = set(keywordsList)
    print(keywordsSet)
    print(len(keywordsSet))

#bow = BagOfWords(keywordsList)
#bagOfWords = bow.generateBagOfWords()
#print(bagOfWords)

# printOrderedKeywords(wordWeight)

pp.themesCount.sort()
print(pp.themesCount)
print(len(pp.themesCount))

nn = NeuralNet(len(keywordsSet), len(pp.themesCount))
nn.createModel()
