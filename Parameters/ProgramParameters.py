# ------------------------------------------ GENERAL PARAMETERS --------------------------------------------------
DATA_FILE_PATH = "C:\\Users\\Chris\\Desktop\\Data\\lfeData.xlsx"
USE_CLI_ARGUMENTS = False      # Enable/Disable the CLI argument parser for overwriting these parameters

# --------------------------------------- PRE PROCESSING PARAMETERS ----------------------------------------------
REMOVE_NUMERIC = True            # Remove any numeric characters or numeric punctuation from the text
REMOVE_SINGLE_LETTERS = True     # Remove any single letters (name abbreviations and prepositions) from the text
REMOVE_KEYWORDS = False          # Remove any listed keywords from the text
REMOVE_EXTRA_SPACES = True       # Remove any extra spaces, new line characters etc from the text

# --------------------------------------- WORD EMBEDDING PARAMETERS ----------------------------------------------
WORD_EMBEDDING_METHOD = 'tf_idf'    # VALID: 'rake', 'text_rank', 'word_count', 'tf_idf'
REMOVE_STOPWORDS = False            # Use the chosen stop word list to purge these words from the text (not for rake)
STEM_TEXT = False                   # Use the chosen stemming algorithm to stem the text (not for rake)

# -------------------------------------- FEATURE CREATION PARAMETERS ---------------------------------------------
# Bag of words parameters
USE_THRESHOLD = False     # Should words/keywords be ignored after a certain threshold
KEYWORD_THRESHOLD = 4     # Value for the threshold at which keywords will be ignored

# ----------------------------------------- CLASSIFIER PARAMETERS ------------------------------------------------
# General Classifier parameters
CLASSIFIER_NAME = 'knn'                 # Type of classifier to use. VALID: 'knn', 'cnb', 'nn', 'svm'
USE_MULTI_LABEL_CLASSIFICATION = True   # Allow multiple labels to be assigned per item
TEST_GROUP_SIZE = 0.25                  # Fraction to split into the test group when performing a test/train split
RANDOM_STATE = None                     # Seed used for random number generation VALID: None, Int

# K-NN parameters
KNN_NEIGHBOURS = 15       # Number of neighbours used when classifying.
KNN_WEIGHTS = 'uniform'     # Determine how distance from neighbours is measured. VALID: 'uniform', 'distance'
KNN_ALGORITHM = 'auto'      # Type of algorithm used. VALID: 'auto', 'ball_tree', 'kd_tree', 'brute'

# Neural Network parameters
NN_BATCH_SIZE = 64         # Number of items to be batched for one training sample
NN_INTERNAL_EPOCHS = 5     # Number of epochs to be performed in a single training fit
NN_BIAS = -5.62            # For setting the bias when training, should be log(posCases/negCases) for dual class

# Support Vector Machine parameters
SVM_KERNEL = 'rbf'            # Which kernel to use for dimensional increase. VALID: 'rbf', 'poly', 'sigmoid', 'linear'
SVM_DEGREE = 3                # Initial degree used in the polynomial kernel (ignored in all other kernels)
SVM_CLASS_WEIGHT = None       # None - no balancing of classes, 'balanced' - automatic proportional weight adjustment
SVM_DECISION_SHAPE = 'ovr'    # Whether to return a "one vs rest" ('ovr') or a "one vs one" ('ovo') decision function

# ------------------------------------------ TESTING PARAMETERS -------------------------------------------------
TEST_RUNS = 1          # Number of tests performed in a single program run
EPOCHS = 1           # Number of iterations within a single test (re-split data, retrain classifier, and re-predict)
PRINT_PROGRESS = True   # Print the current test progress details to the console

# ------------------------------------- RESULTS AND STATS PARAMETERS --------------------------------------------
SAVE_STATS_TO_FILE = False        # Should the resultant statistics be written to a CSV file
SAVE_FILE_NAME = "testStats.csv"   # Filename where results are written (found in "./Output/<SAVE_FILE_NAME>)
