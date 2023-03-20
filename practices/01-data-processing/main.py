"""
01-data-processing
To complete
"""

import glob
import logging
import os
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)


def generate_random_images(n=16, size=(256, 256, 3), save_path="data"):
    """Generate random images for the sake of the exercise"""

    # if data in the folder, don't generate new images
    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        return
        
    elif not os.path.exists(save_path):
        os.mkdir(save_path)

    images = np.random.randint(0, 255, size=(n, *size))
    for image in images:
        path = os.path.join(
            save_path, f"{time.time()}_{np.random.randint(1, 10)}.png")
        cv2.imwrite(path, image)


def get_paths(path):
    """Get all paths finishing by .png (all the images) in the directory"""
    return None


def get_labels(paths):
    """
    Get labels from a path
    format: /path/to/image/imagename_label.png
    label is an integer
    """
    return None

def onehot_labels(labels):
    """
    Convert labels to onehot encoding
    """
    num_classes = 10

    return None

def load_images(paths):
    """
    load, resize, normalize all the images at once
    return a numpy array
    """
    return None

def vis_images(images, labels):
    """
    Visualize images with labels
    you can use matplotlib or opencv
    """
    return None
    

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, 'data')
    logging.info(data_path)

    generate_random_images(save_path=data_path)

    paths = get_paths(data_path)
    labels = get_labels(paths)
    images = load_images(paths)

    logging.info(f"paths: {paths}")
    logging.info(f"labels: {labels}")
    logging.info(f"images: {images}")

    vis_images(images, labels)
