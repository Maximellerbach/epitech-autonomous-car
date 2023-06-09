"""
01-data-processing
"""

import glob
import logging
import os
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)


def generate_random_images(n=16, size=(256, 256, 3), save_path="data"):
    """Generate random images"""

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
    """Get all paths in a directory"""
    return glob.glob(os.path.join(path, '*.png'))


def get_labels(paths):
    """
    Get labels from a path
    format: /path/to/image/imagename_label.png
    """
    return [int(path.split(os.sep)[-1].split('_')[-1].split('.')[0]) for path in paths]


def onehot_labels(labels):
    """
    Convert labels to onehot encoding
    """
    num_classes = 10

    onehot_labels = np.zeros((len(labels), num_classes))

    for i, label in enumerate(labels):
        onehot_labels[i, labels[i]] = 1

    return onehot_labels


def load_images(paths):
    """
    load, resize, normalize all the images at once
    return a numpy array
    """
    images = []
    for path in paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (128, 128))
        image = image.astype('float32') / 255.0
        images.append(image)

    return np.array(images)


def vis_images(images, labels):
    """
    Visualize images with labels
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    num_images = images.shape[0]
    num_rows = int(np.sqrt(num_images))
    num_cols = num_rows

    if num_rows ** 2 < num_images:
        num_rows += 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j

            if index < num_images:
                axes[i, j].imshow(images[index])
                axes[i, j].set_title(labels[index])
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')

    plt.show()


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_path, 'data')
    logging.info(data_path)

    generate_random_images(save_path=data_path)

    paths = get_paths(data_path)
    labels = get_labels(paths)
    labels = onehot_labels(labels)
    images = load_images(paths)

    logging.info(f"paths: {paths}")
    logging.info(f"labels: {labels}")
    logging.info(f"images: {images.shape}")

    vis_images(images, labels)
