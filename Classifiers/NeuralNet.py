from keras import Sequential, optimizers, losses, initializers, callbacks
from keras.layers import Dense, Dropout

from .AbstractClassifier import *


class NeuralNet(AbstractClassifier):

    def __init__(self, featureData, targetData, testSize=0.25, randomState=None, batchSize=32, epochs=10, bias=-5):
        super().__init__(featureData, targetData, testSize, randomState)
        self.inputSize = len(featureData[0])
        self.numThemes = len(ALL_THEMES_LIST)
        self.batchSize = batchSize
        self.epochs = epochs
        self.bias = bias

        self.classifier = self.createModel()
        self.name = "Sequential Neural Network"
        self.y = self.oneHotEncodeTargets(self.y)

    def createModel(self):
        outputBias = initializers.Constant(self.bias)

        model = Sequential()
        model.add(Dense(units=self.inputSize, activation='relu', input_shape=(self.inputSize,)))
        model.add(Dropout(0.5))
        model.add(Dense(units=self.numThemes, activation='sigmoid', bias_initializer=outputBias))

        model.compile(loss=losses.BinaryCrossentropy(), optimizer=optimizers.Adam(lr=1e-3))

        return model

    def train(self):
        super().train()
        self.classifier = self.createModel()

        early_stopping = callbacks.EarlyStopping(monitor='loss', verbose=1, patience=10, mode='max', restore_best_weights=True)

        self.classifier.fit(self.XTrain, self.yTrain, batch_size=self.batchSize, epochs=self.epochs, callbacks=[early_stopping])

    def classifySingleClass(self):
        packagedResults = super().classifySingleClass()
        return [self.getArgMaxIndex(packagedResults[0]), self.oneHotDecodeTargets(packagedResults[1])]

    @staticmethod
    def oneHotEncodeTargets(targets):
        oneHotTargets = []
        for target in targets:
            oneHotMask = []
            for i in range(len(ALL_THEMES_LIST)):
                if i == target:
                    oneHotMask.append(1)
                else:
                    oneHotMask.append(0)
            oneHotTargets.append(oneHotMask)
        return np.array(oneHotTargets)

    @staticmethod
    def oneHotDecodeTargets(yOneHot):
        themeIndexList = []
        for target in yOneHot:
            for i in range(len(target)):
                if target[i] == 1:
                    themeIndexList.append(i)
                    break
        return themeIndexList

    @staticmethod
    def getArgMaxIndex(predictions):
        themeIndexList = []
        for prediction in predictions:
            for i in range(len(prediction)):
                if prediction[i] == max(prediction):
                    themeIndexList.append(i)
                    break
        return themeIndexList
