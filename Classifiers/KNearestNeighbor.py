from sklearn import neighbors, model_selection
import numpy as np


class KNNClassifier:
    def __init__(self, featureData, targetData):
        # Classifier initialisation variables (only mutable on construction)
        self.X = np.array(featureData)  # Convert the feature mask to a numpy array [item, feature mask]
        self.y = np.array(targetData)

        # Single classification variables (mutable on every call of "classify")
        self.actualResults = []  # Empty array pre-reserved for testing set classifications
        self.predictions = []    # Empty array pre-reserved for holding results

    def classify(self, nNeighbours=15, weights='uniform', algorithm='auto', testSize=0.25, randomState=42):
        self.knnClassifier = neighbors.KNeighborsClassifier(nNeighbours, weights=weights, algorithm=algorithm)

        XTrain, XTest, yTrain, yTest = model_selection.train_test_split(self.X,
                                                                        self.y,
                                                                        test_size=testSize,
                                                                        random_state=randomState)

        self.knnClassifier.fit(XTrain, yTrain)
        self.predictions = self.knnClassifier.predict(XTest)
        self.actualResults = yTest
        return [self.predictions, self.actualResults]
