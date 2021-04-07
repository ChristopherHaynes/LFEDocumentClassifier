import pandas as pd
from rake_nltk import Rake

from PreProcessor import *
from WordEmbedding import *
from FeatureCreation import *
from Classifiers import *
from TestManager import *
from StatisticsGenerator import *
from FileIO import *

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

# TODO: [PIPELINE SPLIT 2] - Finish determination of word embedding method
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

# TODO: Add pipeline option for multi theme training (Maybe not required?)
# Encode the target themes into numeric values for classification
for pair in themePairs:
    if USE_MULTI_LABEL_CLASSIFICATION:
        targetMasks.append(encodeThemesToValues(pair[1]))
    else:
        targetMasks.append(encodePrimaryThemeToValue(pair[1]))


# TODO: [PIPELINE SPLIT 4] - Determine which classifier to use and how to initialise it
# Populate "classifier" with the chosen classifier and initialise any hyper-parameters
if CLASSIFIER_NAME == 'knn':
    # TODO: Add multi-label classification to KNN
    classifier = KNNClassifier(featuresMasks, targetMasks, TEST_GROUP_SIZE, RANDOM_STATE, N_NEIGHBOURS, WEIGHTS, ALGORITHM)

elif CLASSIFIER_NAME == 'cnb':
    classifier = ComplementNaiveBayes(featuresMasks, targetMasks, USE_MULTI_LABEL_CLASSIFICATION, TEST_GROUP_SIZE, RANDOM_STATE)

elif CLASSIFIER_NAME == 'nn':
    classifier = NeuralNet(featuresMasks, targetMasks, TEST_GROUP_SIZE, RANDOM_STATE, NN_BATCH_SIZE, NN_INTERNAL_EPOCHS, NN_BIAS)

else:
    print("ERROR - Invalid classifier name chosen")
    breakpoint()

# TODO: [PIPELINE SPLIT 5] - Run tests using the classifier, output results and statistics
for test in range(TEST_RUNS):
    results = runTests(classifier, PRINT_PROGRESS)

    if USE_MULTI_LABEL_CLASSIFICATION:
        testStats = getMultiLabelTestStats(results)
    else:
        testStats = getTestStats(results)

    if PRINT_PROGRESS:
        for name, value in testStats.items():
            print(name + ": " + str(value))

    if SAVE_STATS_TO_FILE:
        writeStatsToFile(testStats, SAVE_FILE_NAME)

pass
