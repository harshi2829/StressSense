import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class SimpleAttention(Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = Dense(input_shape[-1])
        self.U = Dense(input_shape[-1])
        self.V = Dense(1)
        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs: [batch, timesteps, features]
        score = self.V(tf.nn.tanh(self.W(inputs)))
        weights = tf.nn.softmax(score, axis=1)
        context = weights * inputs
        return tf.reduce_sum(context, axis=1)
