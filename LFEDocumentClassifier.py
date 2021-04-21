import pandas as pd
from rake_nltk import Rake

from ArgParser import *
from InputCleaner import *
from WordEmbedding import *
from FeatureCreation import *
from Classifiers import *
from TestManager import *
from FileIO import *

# Handle command line arguments and set program parameters
if USE_CLI_ARGUMENTS:
    args = collectCommandLineArguments()
    CLASSIFIER_NAME = args.classifier
    WORD_EMBEDDING_METHOD = args.wordEmbedding
    TEST_RUNS = args.testRuns
    EPOCHS = args.epochs
    USE_MULTI_LABEL_CLASSIFICATION = args.multiLabel
    SAVE_STATS_TO_FILE = args.save
    SAVE_FILE_NAME = args.fileName
    REMOVE_STOPWORDS = args.removeStopWords
    STEM_TEXT = args.stemText
    KNN_NEIGHBOURS = args.knnNeighbours
    KNN_WEIGHTS = args.knnWeight
    KM_CLUSTERS = args.kmClusters
    KM_N_INIT = args.kmInit
    NN_BATCH_SIZE = args.nnBatchSize
    NN_INTERNAL_EPOCHS = args.nnEpochs
    SVM_KERNEL = args.svmKernel
    SVM_DEGREE = args.svmDegree
    SVM_CLASS_WEIGHT = None if args.svmClassWeight is False else 'balanced'

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
pp = InputCleaner(dataFile, themePairs, GENERATE_1D_THEMES)
pp.cleanText(REMOVE_NUMERIC, REMOVE_SINGLE_LETTERS, REMOVE_KEYWORDS, REMOVE_EXTRA_SPACES)

# TODO: [PIPELINE SPLIT 2] - Finish determination of word embedding method
if WORD_EMBEDDING_METHOD == 'rake':
    r = Rake()
    for i in range(len(themePairs)):
        r.extract_keywords_from_text(themePairs[i][0])
        wordEmbeddings.append(r.get_ranked_phrases_with_scores())

elif WORD_EMBEDDING_METHOD == 'text_rank':
    tr = TextRank(themePairs, REMOVE_STOPWORDS, STEM_TEXT)
    wordEmbeddings = tr.getAllKeywords()

elif WORD_EMBEDDING_METHOD == 'word_count':
    tf = TermFrequency(themePairs, REMOVE_STOPWORDS, STEM_TEXT)
    wordEmbeddings = tf.getAllTermCountsPerDocument()

elif WORD_EMBEDDING_METHOD == 'tf_idf':
    tf = TermFrequency(themePairs, REMOVE_STOPWORDS, STEM_TEXT)
    wordEmbeddings = tf.generateAllTFIDFValues()

else:
    print("ERROR - Invalid Keyword IDing method chosen")
    breakpoint()

# TODO: [PIPELINE SPLIT 3] - Build features from keywords/text
bagOfWords = generateBagOfWords(wordEmbeddings, USE_THRESHOLD, KEYWORD_THRESHOLD)
print("Total Features: " + str(len(bagOfWords)))

# Generate the feature masks which will make up the training features for classification
for scoredPairs in wordEmbeddings:
    featuresMasks.append(generateFeatureMask(bagOfWords, scoredPairs))

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
    classifier = KNNClassifier(featuresMasks, targetMasks,
                               USE_MULTI_LABEL_CLASSIFICATION,
                               TEST_GROUP_SIZE,
                               RANDOM_STATE,
                               KNN_NEIGHBOURS,
                               KNN_WEIGHTS,
                               KNN_ALGORITHM)

elif CLASSIFIER_NAME == 'cnb':
    classifier = ComplementNaiveBayes(featuresMasks, targetMasks,
                                      USE_MULTI_LABEL_CLASSIFICATION,
                                      TEST_GROUP_SIZE,
                                      RANDOM_STATE)

elif CLASSIFIER_NAME == 'nn':
    classifier = NeuralNet(featuresMasks, targetMasks,
                           TEST_GROUP_SIZE,
                           RANDOM_STATE,
                           NN_BATCH_SIZE,
                           NN_INTERNAL_EPOCHS,
                           NN_BIAS)

elif CLASSIFIER_NAME == 'svm':
    classifier = SupportVectorMachine(featuresMasks, targetMasks,
                                      USE_MULTI_LABEL_CLASSIFICATION,
                                      TEST_GROUP_SIZE,
                                      RANDOM_STATE,
                                      SVM_KERNEL,
                                      SVM_DEGREE,
                                      SVM_CLASS_WEIGHT,
                                      SVM_DECISION_SHAPE)

elif CLASSIFIER_NAME == "km":
    classifier = KMeans(featuresMasks, targetMasks,
                        USE_MULTI_LABEL_CLASSIFICATION,
                        TEST_GROUP_SIZE,
                        RANDOM_STATE,
                        KM_CLUSTERS,
                        KM_N_INIT)

else:
    print("ERROR - Invalid classifier name chosen")
    breakpoint()

# TODO: [PIPELINE SPLIT 5] - Run tests using the classifier, output results and statistics
for test in range(TEST_RUNS):
    results = runTests(classifier,
                       EPOCHS,
                       USE_MULTI_LABEL_CLASSIFICATION,
                       CROSS_VALIDATE,
                       CV_FOLDS,
                       PRINT_PROGRESS)

    if CROSS_VALIDATE:
        testStats = results
    else:
        if USE_MULTI_LABEL_CLASSIFICATION:
            testStats = getMultiLabelTestStats(results, EPOCHS)
        else:
            testStats = getTestStats(results, EPOCHS)

    if PRINT_PROGRESS:
        for name, value in testStats.items():
            print(name + ": " + str(value))

    if SAVE_STATS_TO_FILE:
        if CROSS_VALIDATE:
            writeDictionaryToCSV(testStats, SAVE_FILE_NAME, CLASSIFIER_NAME, WORD_EMBEDDING_METHOD, REMOVE_STOPWORDS, STEM_TEXT)
        else:
            writeStatsToFile(testStats, SAVE_FILE_NAME, CLASSIFIER_NAME, WORD_EMBEDDING_METHOD, REMOVE_STOPWORDS, STEM_TEXT)

pass
