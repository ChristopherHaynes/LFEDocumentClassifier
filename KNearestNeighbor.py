from sklearn import neighbors, model_selection
from StatisticsAndResultsGenerator import *
import numpy as np


def getNeighbors(featureMask, targets):
    nNeighbours = 15

    # Convert the feature mask to a numpy array [item, feature mask]
    X = np.array(featureMask)
    y = np.array(targets)

    XTrain, XTest, yTrain, yTest = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)

    knnClassifier = neighbors.KNeighborsClassifier(nNeighbours, weights='uniform')
    knnClassifier.fit(XTrain, yTrain)
    predictions = knnClassifier.predict(XTest)

    correctPredictions = 0
    for i in range(len(predictions)):
        if predictions[i] == yTest[i]:
            correctPredictions = correctPredictions + 1

    percentCorrect = getPercentageCorrect(correctPredictions, len(predictions))

    actualThemesBreakdown = getFractionOfAllThemes(yTest)
    predictionsBreakdown = getFractionOfAllThemes(predictions)
    pass
