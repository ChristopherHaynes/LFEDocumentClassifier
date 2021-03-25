from .AbstractClassifier import *
from sklearn import neighbors


class KNNClassifier(AbstractClassifier):
    def __init__(self,
                 featureData,
                 targetData,
                 testSize=0.25,
                 randomState=None,
                 nNeighbours=15,
                 weights='uniform',
                 algorithm='auto'):
        super().__init__(featureData, targetData, testSize, randomState)
        self.name = "K Nearest Neighbor"
        self.nNeighbours = nNeighbours
        self.weights = weights
        self.algorithm = algorithm

    def train(self):
        super().train()
        self.classifier = neighbors.KNeighborsClassifier(self.nNeighbours, weights=self.weights, algorithm=self.algorithm)
        self.classifier.fit(self.XTrain, self.yTrain)

    def classifySingleClass(self):
        return super().classifySingleClass()
