from Layer import Layer
import numpy as np

class DenseLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backwards(self, output_gradient, learning_rate):
        self.weights -= learning_rate * np.dot(output_gradient, self.input.T)
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)

