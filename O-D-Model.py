"""
O-D-Model.py
    Used for training a model to detect eye anomalies
Bryce Harrington
9/23/2022
"""
import tensorflow as tf
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten

import numpy as np


class ODModel:
    """
    Model handler for O-D
    """
    def __init__(self):
        pass

    def spec_network(self):
        """
        Specify our network architecture and structure
        :sets: self.network
        """
        pass

    def train_model(self):
        """
        Train our defined network into a competent model
        :sets: self.model
        """
        pass

    def inference_model(self):
        """
        Predict with the model saved to the class ( self.model )
        :return: output: output predictions from the model
        """

    @staticmethod
    def inference_model(model_path: str):
        """
        Load and predict with a provided model
        :param model_path: path to the saved model file
        :return: output: output predictions from the model
        """
        pass

