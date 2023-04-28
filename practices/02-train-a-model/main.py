"""
02-train-a-model
author: @maximellerbach
"""

import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def load_data() -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Load the mnist dataset, normalize it, reshape it to your needs and return it
    https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
    """

    return None

def load_data_one_hot() -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    alternative to load_data for the one-hot encoding
    """

    return None


def build_model() -> tf.keras.Model:
    """
    Build a simple CNN model: stack some layers and compile it, you are free to choose the architecture
    https://www.tensorflow.org/api_docs/python/tf/keras/Model

    Be careful with the input shape (28, 28), the output shape and the loss function
    there are two main one for classification: categorical_crossentropy and sparse_categorical_crossentropy

    if you use one-hot encoding for the labels, use categorical_crossentropy
    else use sparse_categorical_crossentropy for integer labels

    (it may be a good exercise to implement both)
    """

    inp = tf.keras.layers.Input(shape=(28, 28, 1))
    
    # TODO: create the model

    # model.compile(
    #     optimizer='adam',
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy']
    # )

    return None


def build_model_one_hot() -> tf.keras.Model:
    """
    alternative to build_model for the one-hot encoding
    """
    inp = tf.keras.layers.Input(shape=(28, 28, 1))

    # TODO: create the model

    # model.compile(
    #     optimizer='adam',
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy']
    # )

    return None


def train_model(
        model: tf.keras.Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        epochs=10,
        batchsize=128,
        **kwargs
) -> tf.keras.Model:
    """
    Train the model using the fit method
    https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

    note that the input data is already reshaped
    """
    return None


def predict(model: tf.keras.Model, x: np.ndarray, sample_size: int):
    """
    Predict the output of the model
    https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict

    take some random samples from the test data and visualize the predictions
    """
    return None


if __name__ == "__main__":
    # load the data
    x_train, y_train, x_test, y_test = load_data()

    # build
    model = build_model()

    # train
    train_model(model, x_train, y_train, x_test, y_test)

    # save the model
    model.save('model.h5')

    # predict
    predict(model, x_test, 5)
