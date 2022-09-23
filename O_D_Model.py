"""
O_D_Model.py
    Used for training a model to detect eye anomalies
Bryce Harrington
9/23/2022
"""
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten
from keras.activations import sigmoid
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
from dataclasses import dataclass


@dataclass
class ODModel:
    """
    Model handler for O-D
    """
    model = None
    callbacks = []
    history = None

    def __init__(self):
        self.spec_network()

    def spec_network(self):
        """
        Specify our network architecture and structure
        :sets: self.model ( even though at this stage it's a network )
        """
        self.model = Sequential()
        self.model.add(Conv2D(input_shape=(512, 512, 3), filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
        self.model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(8, activation='sigmoid'))

        self.model.compile(
            optimizer=Adam(),
            loss=binary_crossentropy,
            metrics=['accuracy']
        )

        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(filepath="./Models/O-D_model.h5")
        self.callbacks = [early_stopping, model_checkpoint]

    def train_model(self, train_data, val_data):
        """
        Train our defined network into a competent model
        :sets: self.model
        """
        self.history = self.model.fit(train_data[0], train_data[1], validation_data=(val_data[0], val_data[1]), epochs=9999, batch_size=16, callbacks=self.callbacks)

    def inference_model(self):
        """
        Predict with the model saved to the class ( self.model )
        :return: output: output predictions from the model
        """
        pass

    @staticmethod
    def inference_model(model_path: str):
        """
        Load and predict with a provided model
        :param model_path: path to the saved model file
        :return: output: output predictions from the model
        """
        pass


if __name__ == "__main__":
    od = ODModel()
