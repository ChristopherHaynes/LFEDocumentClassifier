import pandas as pd
from summa import keywords
from collections import OrderedDict

from PreProcessor import PreProcessor
from TextRank import TextRank
from BagOfWords import BagOfWords
from NeuralNet import NeuralNet

def printOrderedKeywords(wordWeight, number=10):
    nodeWeight = OrderedDict(sorted(wordWeight.items(), key=lambda t: t[1], reverse=True))
    for i, (key, value) in enumerate(nodeWeight.items()):
        print(key + ' - ' + str(value))
        if i > number:
            break

print("Reading File...")
dataFile = pd.read_excel("C:\\Users\\Chris\\Desktop\\Data\\lfeData.xlsx")
print("File Loaded.")

pp = PreProcessor(dataFile)
pp.extractThemePairs()

print("Theme Pairs Length: " + str(len(pp.themePairs)))
print(pp.themePairs[3][0])

tr = TextRank();
themePairKeywords = []
print("Using TextRank to extract keywords.")
for i in range(len(pp.themePairs)):
    textKeywords = []
    wordWeight = tr.getKeywords(pp.themePairs[i][0])
    textKeywords = list(wordWeight.keys())
    themePairKeywords.append(textKeywords)
    
keywordsList = [item for sublist in themePairKeywords for item in sublist]
keywordsSet = set(keywordsList)
print(keywordsSet)
print(len(keywordsSet))

bow = BagOfWords(keywordsList)
bagOfWords = bow.generateBagOfWords()
print(bagOfWords)

#printOrderedKeywords(wordWeight)

pp.themes.sort()
print(pp.themes)
print(len(pp.themes))

nn = NeuralNet(len(keywordsSet), len(pp.themes))
nn.createModel()

