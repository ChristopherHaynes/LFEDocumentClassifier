import pandas as pd
from rake_nltk import Rake

from Preprocessor import PreProcessor
from TextRank import TextRank
from BagOfWords import BagOfWords
from NeuralNet import NeuralNet

# CONSTANTS
KEYWORD_ID_METHOD = 'rake'  # Options: 'rake' 'textrank'

# GLOBAL VARIABLES
themePairs = []  # List of tuples, where the first item contains features and the second contains categories (themes)

# Read raw .XLSX file and store as pandas data-frame
dataFile = pd.read_excel("C:\\Users\\Chris\\Desktop\\Data\\lfeData.xlsx", engine='openpyxl')

# Apply all pre-processing to clean text and themes
# TODO: Add further pipeline options for text cleaning (perhaps controlled with init parameters?)
pp = PreProcessor(dataFile, themePairs)

# TODO: [PIPELINE SPLIT 1] - Determine stop list and stemming method (or disable these options)

# TODO: [PIPELINE SPLIT 2] - Finish determination of keyword IDing method
if KEYWORD_ID_METHOD == 'rake':
    r = Rake()
    for i in range(len(themePairs)):
        r.extract_keywords_from_text(themePairs[i][0])
        themePairs[i][0] = r.get_ranked_phrases_with_scores()

if KEYWORD_ID_METHOD == 'text_rank':
    print("Theme Pairs Length: " + str(len(themePairs)))
    print(themePairs[3][0])

    tr = TextRank()
    themePairKeywords = []
    print("Using TextRank to extract keywords.")
    for i in range(len(themePairs)):
        wordWeight = tr.getKeywords(themePairs[i][0])
        themePairKeywords.append(wordWeight)

    keywordsList = [item for sublist in themePairKeywords for item in sublist]
    keywordsSet = set(keywordsList)
    print(keywordsSet)
    print(len(keywordsSet))

# TODO: [PIPELINE SPLIT 3] - Build features from keywords/text
#bow = BagOfWords(keywordsList)
#bagOfWords = bow.generateBagOfWords()
#print(bagOfWords)

# printOrderedKeywords(wordWeight)

pp.themesCount.sort()
print(pp.themesCount)
print(len(pp.themesCount))

# TODO: [PIPELINE SPLIT 4] - Determine which classifier to use and how to store results
nn = NeuralNet(len(keywordsSet), len(pp.themesCount))
nn.createModel()
