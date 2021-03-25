from .AbstractClassifier import *
from sklearn import naive_bayes


class ComplementNaiveBayes(AbstractClassifier):
    def __init__(self,
                 featureData,
                 targetData,
                 testSize=0.25,
                 randomState=None):
        super().__init__(featureData, targetData, testSize, randomState)
        self.name = "Complement Naive Bayes"

    def train(self):
        super().train()
        self.classifier = naive_bayes.ComplementNB()
        self.classifier.fit(self.XTrain, self.yTrain)

    def classifySingleClass(self):
        return super().classifySingleClass()

    def classifyMultiClass(self):
        self.predictions = self.classifier.predict_log_proba(self.XTest)
        return self.packageResults()
