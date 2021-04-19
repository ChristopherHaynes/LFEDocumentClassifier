from .AbstractClassifier import *
from sklearn import svm


class SupportVectorMachine(AbstractClassifier):

    def __init__(self,
                 featureData,
                 targetData,
                 isMultiLabelClassification=False,
                 testSize=0.25,
                 randomState=None,
                 kernel='rbf',
                 degree=3,
                 classWeight=None,
                 decisionShape='ovr'):
        super().__init__(featureData, targetData, testSize, randomState)
        self.name = "Support Vector Machine"
        self.isMultiLabelClassification = isMultiLabelClassification
        self.kernel = kernel
        self.degree = degree
        self.classWeight = classWeight
        self.decisionShape = decisionShape

    def train(self):
        super().train()
        self.classifier = svm.SVC(kernel=self.kernel,
                                  degree=self.degree,
                                  class_weight=self.classWeight,
                                  decision_function_shape=self.decisionShape)
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
