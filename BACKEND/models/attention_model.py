from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Attention, Dense, Flatten

def build_attention(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Attention()([x, x])
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model
