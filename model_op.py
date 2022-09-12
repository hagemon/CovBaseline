import os
from tensorflow import keras

path = 'model.h5'


def get_model():
    if not os.path.exists(path):
        return None
    return keras.models.load_model(path)


def save_model(model):
    model.save(path)
