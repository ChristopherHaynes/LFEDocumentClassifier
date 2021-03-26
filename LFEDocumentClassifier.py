import pandas as pd
from rake_nltk import Rake

from PreProcessor import *
from WordEmbedding import *
from FeatureCreation import *
from Classifiers import *
from TestManager import *
from StatisticsGenerator import *

# GLOBAL VARIABLES
themePairs = []      # List of tuples, where the first item contains text and the second contains corresponding themes
wordEmbeddings = []  # List of words and their embedded scores per entry (words, keywords, TF-IDF etc)
bagOfWords = []      # List of all the words making up the bag of words (for feature creation)
featuresMasks = []   # Feature mask per entry to match with the bagOfWords structure/order
targetMasks = []     # Target value (class) per entry, aligns with features mask
classifier = None    # Placeholder for the classifier object generated later in the pipeline

# Read raw .XLSX file and store as pandas data-frame
dataFile = pd.read_excel(DATA_FILE_PATH, engine='openpyxl')

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

elif KEYWORD_ID_METHOD == 'word_count':
    tf = TermFrequency(themePairs, REMOVE_STOPWORDS, STEM_TEXT)
    wordEmbeddings = tf.getAllTermCountsPerDocument()

elif KEYWORD_ID_METHOD == 'tf_idf':
    tf = TermFrequency(themePairs, REMOVE_STOPWORDS, STEM_TEXT)
    wordEmbeddings = tf.generateAllTFIDFValues()

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

# TODO: [PIPELINE SPLIT 4] - Determine which classifier to use and how to initialise it
# Populate "classifier" with the chosen classifier and initialise any hyper-parameters
if CLASSIFIER_NAME == 'knn':
    classifier = KNNClassifier(featuresMasks, targetMasks, TEST_SIZE, RANDOM_STATE, N_NEIGHBOURS, WEIGHTS, ALGORITHM)

elif CLASSIFIER_NAME == 'cnb':
    classifier = ComplementNaiveBayes(featuresMasks, targetMasks)

else:
    print("ERROR - Invalid classifier name chosen")
    breakpoint()

# TODO: [PIPELINE SPLIT 5] - Run tests using the classifier, output results and statistics
results = runTests(classifier, PRINT_PROGRESS)

testStats = getTestStats(results)

precisionRecalls = []
correctPercents = []
for result in results:
    correctPercents.append(getAccuracyPercent(result[0], result[1]))
    precisionRecalls.append(getAverageF1Score(result[0], result[1]))

averageRes = sum(correctPercents) / len(correctPercents)
averageF1 = sum(precisionRecalls) / len(precisionRecalls)
print("Average accuracy of " + str(averageRes) + "%")
print("Average F1 of " + str(averageF1))
# TEST_gaussianMixture(featuresMasks, targetMasks)
# TEST_kmeans(featuresMasks, targetMasks)

pass
