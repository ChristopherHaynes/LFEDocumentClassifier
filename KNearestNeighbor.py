from sklearn import neighbors, model_selection
import numpy as np


def getNeighbors(featureMask, targets):
    nNeighbours = 15

    # Convert the feature mask to a numpy array [item, feature mask]
    X = np.array(featureMask)
    y = np.array(targets)

    XTrain, XTest, y_Train, y_Test = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)

    knnClassifier = neighbors.KNeighborsClassifier(nNeighbours, weights='uniform')
    knnClassifier.fit(XTrain, y_Train)
    predictions = knnClassifier.predict(XTest)

    correctPredictions = 0
    for i in range(len(predictions)):
        if predictions[i] == y_Test[i]:
            correctPredictions = correctPredictions + 1

    percentCorrect = (correctPredictions / len(predictions)) * 100
    pass
