import argparse

classifierNames = ['knn', 'cnb', 'nn', 'svm']
wordEmbeddingMethods = ['rake', 'text_rank', 'word_count', 'tf_idf']
knnWeights = ['uniform', 'distance']
svmKernels = ['rbf', 'poly', 'sigmoid', 'linear']


def validateInput(userInput, validOptions):
    # Ensure a valid input has been given
    valid = False
    if userInput in validOptions:
        valid = True

    # If input is invalid, prompt the user to enter a valid option
    while not valid:
        print("\n\nPlease select a valid option from the following list:", validOptions)
        userInput = input('')
        if userInput in validOptions:
            valid = True

    return userInput


def collectCommandLineArguments():
    # Create an argument parser for performing an experiment on the agent
    parser = argparse.ArgumentParser(description='Determining Experiment Parameters')

    # General arguments for determining overall experiment structure
    parser.add_argument('-c', '--classifier',
                        help='The name of the classifier to use' + str(classifierNames))
    parser.add_argument('-we', '--wordEmbedding',
                        help='The type of word embedding to use' + str(wordEmbeddingMethods),
                        default='tf_idf')
    parser.add_argument('-tr', '--testRuns',
                        help='The number test iterations to perform',
                        type=int,
                        default=3)
    parser.add_argument('-e', '--epochs',
                        help='The number of epochs to include in a single test',
                        type=int,
                        default=100)
    parser.add_argument('-ml', '--multiLabel',
                        help='Should each item be given multiple, tiered classifications',
                        default=False,
                        action='store_const',
                        const=True)
    parser.add_argument('-s', '--save',
                        help='Should the results of the test be saved to CSV',
                        default=False,
                        action='store_const',
                        const=True)
    parser.add_argument('-fn', '--fileName',
                        help='The name of the CSV file holding the results',
                        default='testStats.csv')

    # Preprocessing argument flags
    # parser.add_argument('-rn', '--removeNumeric',
    #                         help='Should all numeric characters be removed from the text',
    #                         default=True,
    #                         action='store_const',
    #                         const=False)
    # parser.add_argument('-rc', '--removeSingleChar',
    #                         help='Should any single characters be removed from the text',
    #                         default=False,
    #                         action='store_const',
    #                         const=True)
    # parser.add_argument('-rk', '--removeKeywords',
    #                         help='Should predetermined keywords be removed from the text',
    #                         default=False,
    #                         action='store_const',
    #                         const=True)
    # parser.add_argument('-rs', '--removeSpaces',
    #                         help='Should any extra whitespace be removed from the text',
    #                         default=False,
    #                         action='store_const',
    #                         const=True)

    # Word embedding argument flags
    parser.add_argument('-sw', '--removeStopWords',
                        help='Should stop words be removed from the text',
                        default=False,
                        action='store_const',
                        const=True)
    parser.add_argument('-st', '--stemText',
                        help='Should the text be stemmed (uses porter stemmer)',
                        default=False,
                        action='store_const',
                        const=True)

    # K-NN arguments
    parser.add_argument('-kn', '--knnNeighbours',
                        help='The value of "n" for the KNN algorithm',
                        type=int,
                        default=15)
    parser.add_argument('-kw', '--knnWeight',
                        help='The type of weighting used for the KNN algorithm' + str(classifierNames),
                        default='uniform')

    # NN arguments
    parser.add_argument('-nb', '--nnBatchSize',
                        help='The size of a batch used in one NN training sample',
                        type=int,
                        default=64)
    parser.add_argument('-ne', '--nnEpochs',
                        help='The amount of epochs used in a single NN training fit',
                        type=int,
                        default=5)

    # SVM arguments
    parser.add_argument('-sk', '--svmKernel',
                        help='The type of kernel used for SVM' + str(svmKernels),
                        default='rbf')
    parser.add_argument('-sd', '--svmDegree',
                        help='The initial degree used with the polynomial kernel',
                        type=int,
                        default=3)
    parser.add_argument('-sc', '--svmClassWeight',
                        help='Should the class weights be proportional balanced',
                        default=False,
                        action='store_const',
                        const=True)

    # KM arguments
    parser.add_argument('-mc', '--kmClusters',
                        help='The amount of clusters to fit to',
                        type=int,
                        default=24)
    parser.add_argument('-mi', '--kmInit',
                        help='The number of times to run the algorithm with different centroids',
                        type=int,
                        default=10)

    # Collect the arguments returned by the parser
    args = parser.parse_args()

    # Check validity of all inputs
    args.classifier = validateInput(args.classifier, classifierNames)
    args.wordEmbedding = validateInput(args.wordEmbedding, wordEmbeddingMethods)
    args.knnWeight = validateInput(args.knnWeight, knnWeights)
    args.svmKernel = validateInput(args.svmKernel, svmKernels)

    return args
