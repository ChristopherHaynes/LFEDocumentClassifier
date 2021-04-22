from .AbstractClassifier import *
from sklearn.neural_network import MLPClassifier


class MultiLayerPerceptronSklearn(AbstractClassifier):

    def __init__(self,
                 featureData,
                 targetData,
                 isMultiLabelClassification=False,
                 testSize=0.25,
                 randomState=None):
        super().__init__(featureData, targetData, testSize, randomState)
        self.name = "Multi-layer-perceptron (SKLearn)"
        self.isMultiLabelClassification = isMultiLabelClassification
        self.classifier = MLPClassifier(hidden_layer_sizes=(len(featureData),
                                                            len(featureData),
                                                            len(featureData),
                                                            len(ALL_THEMES_LIST)),
                                        random_state=randomState,
                                        early_stopping=True)

    def train(self):
        super().train()
        self.classifier = MLPClassifier(hidden_layer_sizes=(len(self.X),
                                                            len(self.X),
                                                            len(self.X),
                                                            len(ALL_THEMES_LIST)),
                                        random_state=self.randomState,
                                        early_stopping=True)

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

