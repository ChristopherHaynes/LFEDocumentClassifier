from pathlib import Path
import os
import csv
from sklearn import cluster
from sklearn.metrics import silhouette_score

from .AbstractClassifier import *


def writeScoresToCSV(clusters, score, fileName='kmeanSilhouettes.csv'):
    # If an "Output" directory doesn't exist in the working directory, then create one
    rootPath = Path(__file__).parent
    outputDirPath = (rootPath / "./Output").resolve()
    try:
        os.mkdir(outputDirPath)
    except OSError:
        pass

    # If a results file already exists then open it, otherwise create a new file and write the headers
    filePath = (rootPath / "./Output/" / fileName).resolve()
    isExistingFile = filePath.is_file()
    with open(filePath, 'a', newline='') as file:
        csvWriter = csv.writer(file)
        csvWriter.writerow([str(clusters), str(score)])


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
        clusterLabels = self.classifier.fit_predict(self.XTrain)

        print("Testing Cluster Size: " + str(self.nClusters))
        silhouetteAverage = silhouette_score(self.XTrain, clusterLabels)
        writeScoresToCSV(self.nClusters, silhouetteAverage)

    def classifySingleClass(self):
        return super().classifySingleClass()
