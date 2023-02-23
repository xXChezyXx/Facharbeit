from Activationfunction import Activation
from Layer import Layer
import numpy as np

class ReLU(Activation):#
    def __init__(self):
        
        def relu(x):
            return np.where(x > 0, x, 0)

        def relu_prime(x):
            x = np.where(x > 0, x, 0)
            return np.where(x <= 0, x, 1)
        
        super().__init__(relu, relu_prime)

class Softmax(Layer):#

    def forward(self, input):
        tmp = np.exp(input - np.max(input))
        self.output = tmp / np.sum(tmp)
        return self.output

    def backwards(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)