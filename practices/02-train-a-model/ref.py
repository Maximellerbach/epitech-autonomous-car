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
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # normalize the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # reshape the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return x_train, y_train, x_test, y_test


def load_data_one_hot() -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    alternative to load_data for the one-hot encoding
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # normalize the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # reshape the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # one-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


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
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inp)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.summary()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    """
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)        [(None, 28, 28, 1)]       0

    conv2d (Conv2D)             (None, 26, 26, 32)        320       

    conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496     

    max_pooling2d (MaxPooling2D  (None, 12, 12, 64)       0
    )

    dropout (Dropout)           (None, 12, 12, 64)        0

    flatten (Flatten)           (None, 9216)              0

    dense (Dense)               (None, 128)               1179776

    dropout_1 (Dropout)         (None, 128)               0

    dense_1 (Dense)             (None, 10)                1290

    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    _________________________________________________________________
    """

    ### Number of parameters

    # number of parameters for convolution = (kernel_size * kernel_size * input_channels + 1) * output_channels
    # 3 * 3 * 1 + 1 * 32 = 320
    # 3 * 3 * 32 + 1 * 64 = 18496
    # number of parameters for dense = (input_size + 1) * output_size
    # (9216 + 1) * 128 = 1179776
    # (128 + 1) * 10 = 1290

    ### Number of operations

    # number of operations for convolution = (kernel_size * kernel_size * input_channels) * output_channels * output_width * output_height
    # 3 * 3 * 1 * 32 * 26 * 26 = 194688
    # 3 * 3 * 32 * 64 * 24 * 24 = 10616832
    # number of operations for maxpooling = (kernel_size * kernel_size * input_channels) * output_width * output_height
    # 2 * 2 * 64 * 12 * 12 = 36864
    # number of operations for dense = (input_size + 1) * output_size
    # (9216 + 1) * 128 = 1179776
    # (128 + 1) * 10 = 1290
    # total: 12029450 operations

    return model


def build_model_strides() -> tf.keras.Model:
    """
    alternative to build_model using strides instead of maxpooling
    """

    inp = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inp)
    x = tf.keras.layers.Conv2D(64, 3, strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inp, outputs=out)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    ### Number of parameters same as build_model

    ### Number of operations

    # number of operations for convolution = (kernel_size * kernel_size * input_channels) * output_channels * output_width * output_height
    # 3 * 3 * 1 * 32 * 26 * 26 = 194688
    # 3 * 3 * 32 * 64 * 12 * 12 = 2654208
    # number of operations for dense = (input_size + 1) * output_size
    # (9216 + 1) * 128 = 1179776
    # (128 + 1) * 10 = 1290
    # total: 4031252 operations

    ###############################
    # simply by replacing the maxpooling layer with a stride of 2,
    # we reduced the number of operations by a factor of 3 WITHOUT reducing the number of parameters
    ###############################

    return model


def build_model_one_hot() -> tf.keras.Model:
    """
    alternative to build_model for the one-hot encoding
    """
    inp = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inp)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inp, outputs=out)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


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
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batchsize,
        **kwargs
    )
    return model


def predict(model: tf.keras.Model, x: np.ndarray, sample_size: int):
    """
    Predict the output of the model
    https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict

    take some random samples from the test data and visualize the predictions
    """

    # get random samples
    idx = np.random.randint(0, x.shape[0], sample_size)
    x_sample = x[idx]

    # predict
    y_pred = model.predict(x_sample)

    # visualize
    for i in range(sample_size):
        plt.subplot(1, sample_size, i+1)
        plt.imshow(x_sample[i].reshape(28, 28), cmap='gray')
        plt.title(np.argmax(y_pred[i]))
        plt.axis('off')
    plt.show()


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
