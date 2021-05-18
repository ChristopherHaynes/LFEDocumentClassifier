from .AbstractClassifier import *
from sklearn import naive_bayes


class ComplementNaiveBayes(AbstractClassifier):

    def __init__(self,
                 featureData,
                 targetData,
                 isMultiLabelClassification=False,
                 testSize=0.25,
                 randomState=None,
                 verbose=True):
        super().__init__(featureData, targetData, testSize, randomState)
        self.name = "Complement Naive Bayes"
        self.isMultiLabelClassification = isMultiLabelClassification
        self.verbose = verbose
        self.classifier = naive_bayes.ComplementNB()

    def train(self):
        super().train()
        self.classifier = naive_bayes.ComplementNB()

        if self.isMultiLabelClassification:
            self.classifier.fit(self.XTrain, [item[0] for item in self.yTrain])
        else:
            self.classifier.fit(self.XTrain, self.yTrain)

    def classifySingleClass(self):
        return super().classifySingleClass()

    def classifyMultiClass(self):
        logPredictionValues = self.classifier.predict_log_proba(self.XTest)
        self.predictions = []
        for testNumber in range(len(logPredictionValues[:, 0])):
            topThreeClasses = []
            testResults = logPredictionValues[testNumber]
            for j in range(3):
                for i in range(len(testResults)):
                    if testResults[i] == max(testResults):
                        topThreeClasses.append(i)
                        testResults[i] = min(testResults)
                        break
            self.predictions.append(topThreeClasses)

        return self.packageResults()
