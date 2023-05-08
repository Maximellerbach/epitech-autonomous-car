"""
04-more-training
author: @maximellerbach
"""

import os

import numpy as np
import tensorflow as tf
from ref_utils import DataGenerator, flip, noise

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
    """

    """
    _________________________________________________________________
    Layer (type)                Output Shape              Param #
    =================================================================
    input_1 (InputLayer)        [(None, 120, 160, 3)]     0

    conv2d (Conv2D)             (None, 59, 79, 16)        448

    conv2d_1 (Conv2D)           (None, 29, 39, 32)        4640

    conv2d_2 (Conv2D)           (None, 14, 19, 48)        13872

    conv2d_3 (Conv2D)           (None, 6, 9, 64)          27712

    flatten (Flatten)           (None, 3456)              0

    dense (Dense)               (None, 128)               442496

    dense_1 (Dense)             (None, 64)                8256

    dense_2 (Dense)             (None, 1)                 65        

    =================================================================
    Total params: 497,489
    Trainable params: 497,489
    Non-trainable params: 0
    _________________________________________________________________
    """

    inp = tf.keras.layers.Input(shape=(120, 160, 3))
    x = tf.keras.layers.Conv2D(16, 3, strides=2, activation="relu")(inp)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(48, 3, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, activation="relu")(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    # we are using tanh because we want the output to be between -1 (left) and 1 (right)
    out = tf.keras.layers.Dense(1, activation="tanh")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.summary()

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mse"]
    )

    return model


def train_model(model, datagen: DataGenerator, epochs=10):
    """
    Provide the datagen object to the fit function as source of data.

    (Bonus) It is also a good idea to use a test_datagen to monitor the performance of the model on unseen data.
    In order to do that, you would have to create a new DataGenerator object with a different data directory.
    """
    model.fit(
        datagen,
        steps_per_epoch=len(datagen),
        epochs=epochs,
    )


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


if __name__ == "__main__":
    model = build_model()

    # if you have not implemented any transform funcs yet, just put an empty list []
    datagen = DataGenerator(data_path, [noise, flip])

    train_model(model, datagen, epochs=20)

    model.save("model.h5")
