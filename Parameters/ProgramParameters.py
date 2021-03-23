# ----------------------------------------PRE PROCESSING PARAMETERS ----------------------------------------------
# General Preprocessing parameters
KEYWORD_ID_METHOD = 'text_rank'  # Options: 'rake' 'text_rank' 'none'
REMOVE_NUMERIC = True            # Remove any numeric characters or numeric punctuation from the text
REMOVE_SINGLE_LETTERS = True     # Remove any single letters (name abbreviations and prepositions) from the text
REMOVE_KEYWORDS = False          # Remove any listed keywords from the text
REMOVE_EXTRA_SPACES = True       # Remove any extra spaces, new line characters etc from the text

# Parameters only applying to "Text Rank" and "None" keyword selection
REMOVE_STOPWORDS = True         # Use the chosen stop word list to purge these words from the text
STEM_TEXT = True                # Use the chosen stemming algorithm to stem the text

# ---------------------------------------FEATURE CREATION PARAMETERS ---------------------------------------------
# Bag of words parameters
USE_THRESHOLD = False     # Should words/keywords be ignored after a certain threshold
KEYWORD_THRESHOLD = 4    # Value for the threshold at which keywords will be ignored

# ------------------------------------------CLASSIFIER PARAMETERS ------------------------------------------------
# General Classifier parameters
EPOCHS = 50

# K-NN parameters
N_NEIGHBOURS = 15       # Number of neighbours used when classifying.
WEIGHTS = 'uniform'     # Determine how distance from neighbours is measured. VALID: 'uniform', 'distance'
ALGORITHM = 'auto'      # Type of algorithm used. VALID: 'auto', 'ball_tree', 'kd_tree', 'brute'
TEST_SIZE = 0.25        # Fraction to split into the test group when performing a test/train split
RANDOM_STATE = None     # Seed used for random number generation VALID: None, Int
