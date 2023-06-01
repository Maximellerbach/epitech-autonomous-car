"""
06-performance
author: @maximellerbach
"""

import os

import numpy as np
import tensorflow as tf

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data')


def build_model():
    """Make a multi input / multi output model just for the purpose of explaining how signatures work."""

    # input
    inp_image = tf.keras.layers.Input(shape=(120, 160, 3), name="image")

    # conv layers
    x = tf.keras.layers.SeparableConv2D(24, 3, strides=2, activation="relu")(inp_image)
    x = tf.keras.layers.SeparableConv2D(32, 3, strides=2, activation="relu")(x)
    x = tf.keras.layers.SeparableConv2D(48, 3, strides=2, activation="relu")(x)
    x = tf.keras.layers.SeparableConv2D(96, 3, strides=2, activation="relu")(x)

    # flatten
    x = tf.keras.layers.Flatten()(x)

    # an other input
    inp_speed = tf.keras.layers.Input(shape=(1,), name="speed")
    x = tf.keras.layers.Concatenate()([x, inp_speed])

    # dense layers
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    # outputs
    steering = tf.keras.layers.Dense(1, activation="tanh", name="steering")(x)
    throttle = tf.keras.layers.Dense(1, activation="tanh", name="throttle")(x)

    # model
    model = tf.keras.models.Model(
        inputs=[inp_image, inp_speed],
        outputs=[steering, throttle]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def load_model(model_path):
    """Simply load a .h5 model."""
    return tf.keras.models.load_model(model_path)


def convert_to_tflite(model, model_path):
    """Convert the model to tflite format then save it in the right path."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(model_path, "wb").write(tflite_model)


class TFLiteModel():
    """Load the tflite model and provide a call or predict method."""

    def __init__(self, model_path):
        """
        Load the model and get the default signature runner.

        this might be helpful:
        https://www.tensorflow.org/lite/guide/signatures
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.signatures = self.interpreter.get_signature_list()
        self.runner = self.interpreter.get_signature_runner()

    def __call__(self, **kwargs):
        """Call the runner with the right inputs. Return the output"""
        return self.runner(**kwargs)

    def predict(self, **kwargs):
        return self.__call__(**kwargs)


if __name__ == "__main__":
    model_path = os.path.join(path, "model.h5")
    tflite_model_path = os.path.join(path, "model.tflite")

    # you can also load a model from a .h5 file
    # model = load_model(model_path)
    model = build_model()
    convert_to_tflite(model, tflite_model_path)
    model = TFLiteModel(tflite_model_path)

    # make a prediction (any input will do for the purpose of this example)
    image = np.zeros((1, 120, 160, 3), dtype=np.float32)
    speed = np.array([[0.5]], dtype=np.float32)

    pred = model(image=image, speed=speed)
    steering = pred["steering"]
    throttle = pred["throttle"]

    print("raw prediction:", pred)
    print("steering:", steering)
    print("throttle:", throttle)
