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
    """
    Make a multi input / multi output model. No need to train it.
    This is just for the purpose of playing with inputs/outputs and see how signatures work.

    for example: 
        input image: (120, 160, 3)
        input speed: (1,)

        output steering: (1,)
        output throttle: (1,)
    """

    # model = tf.keras.models.Model(
    #     inputs=[inp_image, inp_speed],
    #     outputs=[steering, throttle]
    # )
    # model.compile(optimizer="adam", loss="mse")

    # return model
    pass


def load_model(model_path):
    """Simply load a .h5 model."""
    pass


def convert_to_tflite(model, model_path):
    """Convert the model to tflite format then save it in the right path."""
    pass


class TFLiteModel():
    """Load the tflite model and provide a call or predict method."""

    def __init__(self, model_path):
        """
        Load the model and get the default signature runner.

        this might be helpful:
        https://www.tensorflow.org/lite/guide/signatures#python
        """
        pass

    def __call__(self, **kwargs):
        pass


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
