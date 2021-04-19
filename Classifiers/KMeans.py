from sklearn import cluster
from .AbstractClassifier import *


class KMeans(AbstractClassifier):

    def __init__(self,
                 featureData,
                 targetData,
                 useMultiLabelClassification,
                 testSize=0.25,
                 randomState=None,
                 nClusters=15,
                 nInit=10):
        super().__init__(featureData, targetData, testSize, randomState)
        self.name = "K Means"
        self.useMultiLabelClassification = useMultiLabelClassification
        self.nClusters = nClusters
        self.nInit = nInit

    def train(self):
        super().train()
        self.classifier = cluster.KMeans(n_clusters=self.nClusters, n_init=self.nInit)
        self.classifier.fit(self.XTrain)
        print("N-Clusters = " + str(self.nClusters) + ". Train Set Score = " + str(self.classifier.score(self.XTrain)))
        print("N-Clusters = " + str(self.nClusters) + ". Test Set Score = " + str(self.classifier.score(self.XTest)))

    def classifySingleClass(self):
        return super().classifySingleClass()


