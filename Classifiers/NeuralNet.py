from keras import Sequential
from keras.layers import Dense

from Parameters import ALL_THEMES_LIST
from .AbstractClassifier import *


class NeuralNet(AbstractClassifier):
    def __init__(self, featureData, targetData, testSize=0.25, randomState=None):
        super().__init__(featureData, targetData, testSize, randomState)
        self.inputSize = len(featureData[0])
        self.numThemes = len(ALL_THEMES_LIST)
        self.classifier = self.createModel()
        self.name = "Sequential Neural Network"
        self.oneHotEncodeTargets()
    
    def createModel(self):
        model = Sequential()
        model.add(Dense(units=self.inputSize, activation='relu', input_shape=(self.inputSize,)))
        model.add(Dense(units=self.inputSize, activation='relu'))
        model.add(Dense(units=self.numThemes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.summary()
        return model

    def oneHotEncodeTargets(self):
        oneHotTargets = []
        for target in self.y:
            oneHotMask = []
            for i in range(len(ALL_THEMES_LIST)):
                if i == target:
                    oneHotMask.append(1)
                else:
                    oneHotMask.append(0)
            oneHotTargets.append(oneHotMask)
        self.y = np.array(oneHotTargets)

    def train(self):
        super().train()
        self.classifier.fit(self.XTrain, self.yTrain)

    # TODO: Re-encode yTest from one-hot to theme index and take argMax index from predictions
    def classifySingleClass(self):
        return super().classifySingleClass()


