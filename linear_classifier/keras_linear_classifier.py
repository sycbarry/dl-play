"""
Building a linear classifier using Keras
"""

from tensorflow import keras
import numpy as np


"""
This is the definition of a layer that we can use in deep learning model
again, a deep learning mdoel is just a definition of a lot of layers stacked together
that we pass data through and "train"
"""
class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        """
        again the linear classifier is fundamentally a linear equation.
        """
        input_dim = input_shape[-1]
        self.W = self.add_weight(
            shape=(input_dim, self.units), initializer="random_normal"
        )
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal"))

    def call(self, inputs): 
        """
        this is the defined forward pass computation
        """
        y = tf.matmul(inputs, self.W) + self.b 
        if self.activation is not None: 
            y = self.activation(y)
        return y




    


