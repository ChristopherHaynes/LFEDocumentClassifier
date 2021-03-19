import pandas as pd
from rake_nltk import Rake

from Preprocessor import PreProcessor
from TextRank import TextRank
from FeatureCreation import *

# GLOBAL CONSTANTS
KEYWORD_ID_METHOD = 'rake'  # Options: 'rake' 'text_rank'
REMOVE_NUMERIC = True            # Remove any numeric characters or numeric punctuation from the text
REMOVE_SINGLE_LETTERS = True     # Remove any single letters (name abbreviations and prepositions) from the text
REMOVE_KEYWORDS = False          # Remove any listed keywords from the text
REMOVE_EXTRA_SPACES = True       # Remove any extra spaces, new line characters etc from the text

# GLOBAL VARIABLES
themePairs = []      # List of tuples, where the first item contains text and the second contains corresponding themes
wordEmbeddings = []  # List of words and their embedded scores per entry (words, keywords, TF-IDF etc)
bagOfWords = []      # List of all the words making up the bag of words (for feature creation)
featuresMasks = []    # Feature mask per entry to match with the bagOfWords structure/order

# Read raw .XLSX file and store as pandas data-frame
dataFile = pd.read_excel("C:\\Users\\Chris\\Desktop\\Data\\lfeData.xlsx", engine='openpyxl')

# TODO: [PIPELINE SPLIT 1] - Determine stop list and stemming method (or disable these options)
# Apply all pre-processing to clean text and themes
pp = PreProcessor(dataFile, themePairs)
pp.cleanText(REMOVE_NUMERIC, REMOVE_SINGLE_LETTERS, REMOVE_KEYWORDS, REMOVE_EXTRA_SPACES)

# TODO: [PIPELINE SPLIT 2] - Finish determination of keyword IDing method
if KEYWORD_ID_METHOD == 'rake':
    r = Rake()
    for i in range(len(themePairs)):
        r.extract_keywords_from_text(themePairs[i][0])
        wordEmbeddings.append(r.get_ranked_phrases_with_scores())

elif KEYWORD_ID_METHOD == 'text_rank':
    tr = TextRank(themePairs)
    for i in range(len(themePairs)):
        wordWeight = tr.getKeywords(themePairs[i][0])
        wordEmbeddings.append(wordWeight)

else:
    print("ERROR - Invalid Keyword IDing method chosen")
    breakpoint()

# TODO: [PIPELINE SPLIT 3] - Build features from keywords/text
bagOfWords = generateBagOfWords(wordEmbeddings)
print(len(bagOfWords))

for scoredPairs in wordEmbeddings:
    featuresMasks.append(generateFeatureMask(bagOfWords, scoredPairs))

# TODO: [PIPELINE SPLIT 4] - Determine which classifier to use and how to store results
pass
