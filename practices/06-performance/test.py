try:
    import ref as main
except ImportError:
    import main

import os
import unittest

import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
tflite_path = os.path.join(path, 'model.tflite')

class TestTraining(unittest.TestCase):
    def tearDownClass():
        if os.path.exists(tflite_path):
            os.remove(tflite_path)

    def test_tflite_model(self):
        model = main.build_model()
        main.convert_to_tflite(model, tflite_path)

        # load the model
        tflite_model = main.TFLiteModel(tflite_path)

        # test with random float32 inputs
        image = np.random.random((1, 120, 160, 3)).astype(np.float32)
        speed = np.random.random((1, 1)).astype(np.float32)

        pred = tflite_model(image=image, speed=speed)
        steering = pred["steering"]
        throttle = pred["throttle"]

        self.assertTrue(-1 <= steering <= 1)
        self.assertTrue(-1 <= throttle <= 1)


if __name__ == '__main__':
    unittest.main()
