import numpy as np
from tensorflow import keras

n_classes = 10
input_shape = (28*28)


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.reshape(x_train, (60000, 28 * 28))
    x_test = np.reshape(x_test, (10000, 28 * 28))

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    return x_train, x_test, y_train, y_test
