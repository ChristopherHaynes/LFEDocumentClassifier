# ------------------------------------------ GENERAL PARAMETERS --------------------------------------------------
DATA_FILE_PATH = "C:\\Users\\Chris\\Desktop\\Data\\lfeData.xlsx"

# --------------------------------------- PRE PROCESSING PARAMETERS ----------------------------------------------
REMOVE_NUMERIC = True            # Remove any numeric characters or numeric punctuation from the text
REMOVE_SINGLE_LETTERS = True     # Remove any single letters (name abbreviations and prepositions) from the text
REMOVE_KEYWORDS = False          # Remove any listed keywords from the text
REMOVE_EXTRA_SPACES = True       # Remove any extra spaces, new line characters etc from the text

# --------------------------------------- WORD EMBEDDING PARAMETERS ----------------------------------------------
KEYWORD_ID_METHOD = 'tf_idf'    # VALID: 'rake', 'text_rank', 'word_count', 'tf_idf'
REMOVE_STOPWORDS = False            # Use the chosen stop word list to purge these words from the text (not for rake)
STEM_TEXT = False                   # Use the chosen stemming algorithm to stem the text (not for rake)

# -------------------------------------- FEATURE CREATION PARAMETERS ---------------------------------------------
# Bag of words parameters
USE_THRESHOLD = False     # Should words/keywords be ignored after a certain threshold
KEYWORD_THRESHOLD = 4     # Value for the threshold at which keywords will be ignored

# ----------------------------------------- CLASSIFIER PARAMETERS ------------------------------------------------
# General Classifier parameters
CLASSIFIER_NAME = 'cnb'     # Type of classifier to use. VALID: 'knn', 'cnb'
TEST_GROUP_SIZE = 0.25        # Fraction to split into the test group when performing a test/train split
RANDOM_STATE = None     # Seed used for random number generation VALID: None, Int

# K-NN parameters
N_NEIGHBOURS = 15       # Number of neighbours used when classifying.
WEIGHTS = 'uniform'     # Determine how distance from neighbours is measured. VALID: 'uniform', 'distance'
ALGORITHM = 'auto'      # Type of algorithm used. VALID: 'auto', 'ball_tree', 'kd_tree', 'brute'

# ------------------------------------------ TESTING PARAMETERS -------------------------------------------------
TEST_RUNS = 5           # Number of tests performed in a single program run
EPOCHS = 200            # Number of iterations within a single test (re-split data, retrain classifier, and re-predict)
PRINT_PROGRESS = True   # Print the current test progress details to the console

# ------------------------------------- RESULTS AND STATS PARAMETERS --------------------------------------------
SAVE_STATS_TO_FILE = True          # Should the resultant statistics be written to a CSV file
SAVE_FILE_NAME = "testStats.csv"   # Filename to write the results to (found in "./Output/<SAVE_FILE_NAME>)
