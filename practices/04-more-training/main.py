"""
04-more-training
author: @maximellerbach
"""

import glob
import os

import cv2
import numpy as np
import tensorflow as tf
# also imports your data-augmentation functions here
from utils import DataGenerator

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data')


def build_model():
    """
    Build a simple CNN model: stack some layers and compile it, please mind the size of the model.
    It might also be a good idea to try to reduce as much as possible the number of operations !
    a quick reminder: https://www.tensorflow.org/api_docs/python/tf/keras/Model

    Be careful with the input shape (120, 160, 3).
    Note that we want to use the activation function "tanh" on the output layer.

    If you are brave enough, you can try to add more inputs/outputs BUT you will also have to change the DataGenerator class to match those.

    For the model compilation, use adam optimizer and mse loss.
    """

    pass


def train_model(model, datagen: DataGenerator, epochs=10):
    """
    Provide the datagen object to the fit function as source of data.

    (Bonus) It is also a good idea to use a test_datagen to monitor the performance of the model on unseen data.
    In order to do that, you would have to create a new DataGenerator object with a different data directory.
    """
    pass


def load_model(model_path):
    """"Simply load the model."""
    pass


def predict(model, data_path):
    """
    Predict the steering angle for each image in the data_path.
    You can sort the images by name (date) to get the correct order then play the images as a video.

    hint: you can use cv2 to display the images
    You can also draw a visualisation of the steering angle on the image.
    """

    pass


if __name__ == "__main__":
    # model = load_model("model.h5")
    model = build_model()

    # if you have not implemented any transform funcs yet, just put an empty list []
    datagen = DataGenerator(data_path, [], batch_size=32)

    # if the traning takes too much time, you can try to reduce the batch_size and the number of epochs
    train_model(model, datagen, epochs=5)

    model.save("trained_model.h5")
    predict(model, data_path)
