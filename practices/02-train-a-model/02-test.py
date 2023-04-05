try:
    import ref as main
except ImportError:
    import main

import unittest
import numpy as np


class TestTraining(unittest.TestCase):
    def test_build_model(self):
        model = main.build_model()
        self.assertEqual(model.input_shape, (None, 28, 28, 1))
        self.assertEqual(model.output_shape, (None, 10))

    def test_load_data(self):
        x_train, y_train, x_test, y_test = main.load_data()
        self.assertEqual(x_train.shape, (60000, 28, 28, 1))
        self.assertEqual(y_train.shape, (60000,))
        self.assertEqual(x_test.shape, (10000, 28, 28, 1))
        self.assertEqual(y_test.shape, (10000,))

    def test_train_model(self):
        model = main.build_model()
        x_train = np.random.random((100, 28, 28, 1))
        y_train = np.random.randint(0, 10, (100, 1))
        x_test = np.random.random((100, 28, 28, 1))
        y_test = np.random.randint(0, 10, (100, 1))

        main.train_model(model, x_train, y_train, x_test, y_test, epochs=1)

if __name__ == '__main__':
    unittest.main()