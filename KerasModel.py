#Supporting Libraries
import numpy as np
import pandas as pd

#Keras
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import EarlyStopping

#Model
class Model():
    def __init__(self, predictors, targets):
        self.model = Sequential()
        self.model_name = ''

        self.targets = to_categorical(targets)
        self.predictors = predictors.as_matrix()
        self.predictors = np.multiply(self.predictors, 1/255)
        
    def build_model(self, layers = 1, nodes = 10):          
        #layers
        for layer in range(layers):
            if layer==0:
                self.model.add(Dense(nodes, activation='relu', input_shape=(self.predictors.shape[1],)))
            else:
                self.model.add(Dense(nodes, activation='relu'))
            
        #Output Layer
        self.model.add(Dense(self.targets.shape[1], activation='softmax'))
        
        self.model_name = 'model {}layers and {}nodes.h5'.format(layers, nodes)
        print('Model has been created, ready for training!')
        
    def train_model(self, patience = 1, epochs = 10, val_predictors=0, val_targets=0):        
        #Compile
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        #Training
        early_stopping_monitor = EarlyStopping(patience=patience)
        val_targets = to_categorical(val_targets)
        
        self.model.fit(self.predictors, self.targets, validation_data=(val_predictors, val_targets), epochs=epochs, callbacks=[early_stopping_monitor])
        
        print(self.model.summary())
        print('Model has been trained, validated, and saved!')
        
        self.model.save('Trained Models/{}'.format(self.model_name))
        

