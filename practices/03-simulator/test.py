try:
    import ref as main
except ImportError:
    import main

import unittest
import os
import glob
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data')


class TestDataProcessing(unittest.TestCase):
    def setUpClass():
        if os.path.exists(data_path):
            paths = glob.glob(os.path.join(data_path, '*'))
            for path in paths:
                os.remove(path)

    def test_generate_random_images(self):
        main.generate_random_images(16, save_path=data_path)
        self.assertEqual(len(os.listdir(data_path)), 16)

    def test_get_paths(self):
        paths = main.get_paths(data_path)
        self.assertEqual(len(paths), 16)

    def test_get_labels(self):
        paths = main.get_paths(data_path)
        labels = main.get_labels(paths)
        self.assertEqual(len(labels), 16)

        onehot_labels = main.onehot_labels(labels)
        self.assertEqual(onehot_labels.shape, (16, 10))

    def test_load_images(self):
        paths = main.get_paths(data_path)
        images = main.load_images(paths)
        self.assertEqual(images.shape, (16, 128, 128, 3))
        self.assertLessEqual(np.max(images), 1.0)


if __name__ == '__main__':
    unittest.main()