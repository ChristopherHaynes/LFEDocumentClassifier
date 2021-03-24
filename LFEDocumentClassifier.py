import pandas as pd
from rake_nltk import Rake

from Preprocessor import PreProcessor
from TextRank import TextRank
from FeatureCreation import *
from Classifiers import *
from Parameters import *
from StatisticsAndResultsGenerator import *

# GLOBAL VARIABLES
themePairs = []      # List of tuples, where the first item contains text and the second contains corresponding themes
wordEmbeddings = []  # List of words and their embedded scores per entry (words, keywords, TF-IDF etc)
bagOfWords = []      # List of all the words making up the bag of words (for feature creation)
featuresMasks = []   # Feature mask per entry to match with the bagOfWords structure/order
targetMasks = []     # Target value (class) per entry, aligns with features mask

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
    wordEmbeddings = tr.getAllKeywords()

elif KEYWORD_ID_METHOD == 'none':
    processedPairs = PreProcessor.splitOnSentenceAndWords(themePairs)
    if REMOVE_STOPWORDS:
        processedPairs = PreProcessor.removeStopWords(processedPairs)
    if STEM_TEXT:
        processedPairs = PreProcessor.stemText(processedPairs)
    if EMBEDDING_SCORE_METHOD == 'word_count':
        for pair in processedPairs:
            wordEmbeddings.append(generateTermCountList(pair[0]))
    elif EMBEDDING_SCORE_METHOD == 'tf_idf':
        wordEmbeddings = generateAllTFIDFValues(processedPairs)
    else:
        print("ERROR - Invalid Embedding method chosen")
        breakpoint()
else:
    print("ERROR - Invalid Keyword IDing method chosen")
    breakpoint()

# TODO: [PIPELINE SPLIT 3] - Build features from keywords/text
bagOfWords = generateBagOfWords(wordEmbeddings, USE_THRESHOLD, KEYWORD_THRESHOLD)
print(len(bagOfWords))

# Generate the feature masks which will make up the training features for classification
for scoredPairs in wordEmbeddings:
    featuresMasks.append(generateFeatureMask(bagOfWords, scoredPairs))

# Encode the target themes into numeric values for classification
for pair in themePairs:
    targetMasks.append(encodePrimaryThemeToValue(pair[1]))

# TODO: [PIPELINE SPLIT 4] - Determine which classifier to use and how to store results
classifier = KNNClassifier(featuresMasks, targetMasks)

averageResults = []
for test in range(0, 10):
    correctPercents = []
    for epoch in range(0, EPOCHS):
        results = TEST_naiveBayesMultinomial(featuresMasks, targetMasks)
        correctPercents.append(getPercentageCorrect(results[0], results[1]))

    averageRes = sum(correctPercents) / len(correctPercents)
    averageResults.append(averageRes)
    print("Test " + str(test + 1) + " average accuracy of " + str(averageRes) + "%")

# TEST_gaussianMixture(featuresMasks, targetMasks)
# TEST_kmeans(featuresMasks, targetMasks)

pass
