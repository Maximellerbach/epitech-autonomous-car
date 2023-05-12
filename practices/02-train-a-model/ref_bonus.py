import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


def predict_from_path(image_path, model) -> int:
    """
    Load, resize, reshape, normalize the image and predict the class.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    x = image.reshape(1, 28, 28, 1) # add batch dimension and grayscale channel

    # normalize the data
    x = x / 255.0

    # predict
    pred = model.predict(x) # <- [0, 0, ..., 1.0]
    prediction = np.argmax(pred)

    plt.imshow(image)
    plt.title(f"Prediction: {prediction}")
    plt.show()

if __name__ == "__main__":
    # load the model
    model = tf.keras.models.load_model("model.h5")

    # predict
    predict_from_path("test.png", model)

