from keras import Sequential
from keras.layers import Dense

class NeuralNet:
    def __init__(self, inputSize, numThemes):        
        self.inputSize = inputSize
        self.numThemes = numThemes
        self.model = Sequential()
    
    def createModel(self):      
        self.model.add(Dense(units = inputSize, activation='relu', input_dim=inputSize))
        self.model.add(Dense(units = inputSize, activation='relu'))
        self.model.add(Dense(units = numThemes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])
        print("Created NN.")