try:
    import ref as main
except ImportError:
    import main

import unittest
import os
import glob

path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(path, 'data')


class Test(unittest.TestCase):
    def setUpClass():
        if os.path.exists(data_path):
            paths = glob.glob(os.path.join(data_path, '*'))
            for path in paths:
                os.remove(path)

    def test_generate_random_images(self):
        main.generate_random_images(10, save_path=data_path)
        self.assertEqual(len(os.listdir(data_path)), 10)

    def test_get_paths(self):
        paths = main.get_paths(data_path)
        self.assertEqual(len(paths), 10)

    def test_get_labels(self):
        paths = main.get_paths(data_path)
        labels = main.get_labels(paths)
        self.assertEqual(len(labels), 10)

    def test_load_images(self):
        paths = main.get_paths(data_path)
        images = main.load_images(paths)
        self.assertEqual(images.shape, (10, 128, 128, 3))
