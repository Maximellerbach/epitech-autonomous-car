"""
01-data-processing
"""

import os
import matplotlib
import glob
import cv2
import time

import numpy as np

import logging

logging.basicConfig(level=logging.DEBUG)


def generate_random_images(n=10, size=(256, 256, 3), save_path="data"):
    """Generate random images"""

    # if data in the folder, don't generate new images
    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        return

    images = np.random.randint(0, 255, size=(n, *size))
    for image in images:
        path = os.path.join(
            save_path, f"{time.time()}_{np.random.randint(1, 10)}.png")
        cv2.imwrite(path, image)


def get_paths(path):
    """Get all paths in a directory"""
    return glob.glob(os.path.join(path, '*.png'))


def get_labels(paths):
    """
    Get labels from a path
    format: /path/to/image/imagename_label.png
    """
    return [int(path.split(os.sep)[-1].split('.')[0]) for path in paths]


def load_images(paths):
    """
    load, resize, normalize the images
    return a numpy array
    """
    images = []
    for path in paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (128, 128))
        image = image.astype('float32') / 255.0
        images.append(image)

    return np.array(images)


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, 'data')
    logging.debug(data_path)

    generate_random_images(10, save_path=data_path)

    paths = get_paths(data_path)
    labels = get_labels(paths)
    images = load_images(paths)

    logging.debug(f"paths: {paths}")
    logging.debug(f"labels: {labels}")
    logging.debug(f"images: {images.shape}")
