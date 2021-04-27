from sklearn.model_selection import cross_validate, train_test_split
import numpy as np

from Parameters import ALL_THEMES_LIST


class AbstractClassifier:

    def __init__(self, featureData, targetData, testSize=0.25, randomState=None):
        # Classifier initialisation variables (only mutable on construction)
        self.name = "ABSTRACT"          # Classifier name used for reference in results
        self.X = np.array(featureData)  # Convert the feature mask to a numpy array [item, feature mask]
        self.y = np.array(targetData)   # Convert the target labels to a numpy array [item, feature mask]
        self.testSize = testSize        # float representing the fraction of data to be split into test
        self.randomState = randomState  # Int or None to seed random number generator

        # Single classification variables (mutable after construction)
        self.classifier = None  # Placeholder for classifier to be built into in concrete classes
        self.XTrain = []        # Training data input
        self.XTest = []         # Test data input
        self.yTrain = []        # Training data labels (classes)
        self.yTest = []         # Test data labels (classes)
        self.predictions = []   # Classifier predictions for XTest

    def splitTestTrainData(self):
        self.XTrain, self.XTest, self.yTrain, self.yTest = \
            train_test_split(self.X, self.y, test_size=self.testSize, random_state=self.randomState)

    def crossValidate(self, cv):
        scoringMetrics = ['accuracy', 'precision_micro', 'precision_macro', 'recall_micro', 'recall_macro',
                          'f1_micro', 'f1_macro']
        cvResults = cross_validate(self.classifier, self.X, self.y, cv=cv, scoring=scoringMetrics)

        # Take the average of each of the scoring metrics, package them together and return
        averageResults = dict()
        for metric in scoringMetrics:
            metricName = "test_" + metric
            averageResults[metric] = sum(cvResults[metricName]) / len(cvResults[metricName])

        # Keep the fit and "score" times
        averageResults['fit_time'] = sum(cvResults['fit_time']) / len(cvResults['fit_time'])
        averageResults['score_time'] = sum(cvResults['score_time']) / len(cvResults['score_time'])

        return averageResults

    def train(self):
        if len(self.XTrain) == 0:
            self.splitTestTrainData()

    def classifySingleClass(self):
        self.predictions = self.classifier.predict(self.XTest)
        return self.packageResults()

    def classifyMultiClass(self):
        pass

    def validate(self):
        # TODO: Consider how validation dataset should be run/returned
        pass

    def packageResults(self):
        return [self.predictions, self.yTest]
