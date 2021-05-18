from .AbstractClassifier import *
from sklearn.neural_network import MLPClassifier


class MultiLayerPerceptronSklearn(AbstractClassifier):

    def __init__(self,
                 featureData,
                 targetData,
                 categoryCount,
                 isMultiLabelClassification=False,
                 testSize=0.25,
                 randomState=None,
                 verbose=True):
        super().__init__(featureData, targetData, testSize, randomState)
        self.name = "Multi-layer-perceptron (SKLearn)"
        self.categoryCount = categoryCount
        self.isMultiLabelClassification = isMultiLabelClassification
        self.verbose = verbose
        self.classifier = MLPClassifier(hidden_layer_sizes=(int((len(featureData) + categoryCount) / 2)),
                                        random_state=randomState,
                                        early_stopping=True,
                                        verbose=self.verbose)

    def train(self):
        super().train()
        self.classifier = MLPClassifier(hidden_layer_sizes=(int((len(self.XTrain) + len(self.categoryCount)) / 2)),
                                        random_state=self.randomState,
                                        early_stopping=True,
                                        verbose=self.verbose)

        if self.isMultiLabelClassification:
            # TODO: SKLEARN will handle multi label, y must be in the format (n_samples, n_outputs) - vector encoded
            pass
        else:
            self.classifier.fit(self.XTrain, self.yTrain)

    def classifySingleClass(self):
        return super().classifySingleClass()

    def classifyMultiClass(self):
        # TODO: finish the multi-label approach
        pass

