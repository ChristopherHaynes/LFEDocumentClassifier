# General Classifier Constants
EPOCHS = 100

# K-NN Constants
N_NEIGHBOURS = 15       # Number of neighbours used when classifying.
WEIGHTS = 'uniform'     # Determine how distance from neighbours is measured. VALID: 'uniform', 'distance'
ALGORITHM = 'auto'      # Type of algorithm used. VALID: 'auto', 'ball_tree', 'kd_tree', 'brute'
TEST_SIZE = 0.25        # Fraction to split into the test group when performing a test/train split
RANDOM_STATE = None     # Seed used for random number generation VALID: None, Int
