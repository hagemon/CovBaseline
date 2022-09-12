from data_op import input_shape, n_classes
from tensorflow import keras
from tensorflow.keras import layers


def train(x_train, x_test, y_train, y_test):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(n_classes, activation='softmax'),
        ]
    )

    model.summary()

    batch_size = 128
    epochs = 15

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        batch_size=batch_size, epochs=epochs,
        validation_split=0.1
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return model
