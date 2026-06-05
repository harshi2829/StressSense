from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def build_cnn(input_shape):
    model = Sequential()

    model.add(Conv1D(
        32, 3,
        activation="relu",
        padding="same",
        input_shape=input_shape
    ))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(
        64, 3,
        activation="relu",
        padding="same"
    ))
    model.add(MaxPooling1D(2))

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    return model
