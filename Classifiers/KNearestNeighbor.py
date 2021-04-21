from .AbstractClassifier import *
from sklearn import neighbors
from skmultilearn.adapt import MLkNN


class KNNClassifier(AbstractClassifier):

    def __init__(self,
                 featureData,
                 targetData,
                 useMultiLabelClassification,
                 testSize=0.25,
                 randomState=None,
                 nNeighbours=15,
                 weights='uniform',
                 algorithm='auto'):
        super().__init__(featureData, targetData, testSize, randomState)
        self.name = "K Nearest Neighbor"
        self.useMultiLabelClassification = useMultiLabelClassification
        self.nNeighbours = nNeighbours
        self.weights = weights
        self.algorithm = algorithm
        if self.useMultiLabelClassification:
            self.classifier = MLkNN(self.nNeighbours)
        else:
            self.classifier = neighbors.KNeighborsClassifier(self.nNeighbours, weights=self.weights, algorithm=self.algorithm)

    def train(self):
        super().train()
        if self.useMultiLabelClassification:
            self.classifier = MLkNN(self.nNeighbours)
            self.classifier.fit(self.XTrain, self.encodeThemeIndicesToMatrix(self.yTrain))
        else:
            self.classifier = neighbors.KNeighborsClassifier(self.nNeighbours, weights=self.weights, algorithm=self.algorithm)
            self.classifier.fit(self.XTrain, self.yTrain)

    def classifySingleClass(self):
        return super().classifySingleClass()

    # TODO: Confusing approach (calling single class in abstract) consider refactor (also look into spacey sparse matrix)
    def classifyMultiClass(self):
        packagedResults = super().classifySingleClass()
        return [packagedResults[0].rows, packagedResults[1]]

    @staticmethod
    def encodeThemeIndicesToMatrix(allTargetsList):
        targetsMatrix = []
        for targets in allTargetsList:
            targetMask = np.zeros(len(ALL_THEMES_LIST))
            for target in targets:
                targetMask[target] = 1
            targetsMatrix.append(targetMask)
        return np.array(targetsMatrix)

    @staticmethod
    def decodeMatrixToThemeIndices(resultsMatrix):
        themeIndexList = []
        for result in resultsMatrix:
            themeIndexes = []
            for i in range(len(result)):
                if result[i] == 1:
                    themeIndexes.append(i)
            themeIndexList.append(themeIndexes)
        return themeIndexList
