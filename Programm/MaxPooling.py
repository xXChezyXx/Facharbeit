from Layer import Layer
import numpy as np
import math
import matplotlib.pyplot as plt

class MaxPooling(Layer):

    def __init__(self, input_shape):
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (input_depth, math.ceil(input_height/2), math.ceil(input_width/2))
        self.kernels_shape = (input_depth, input_depth, 2, 2)
        self.input_max = np.zeros(self.input_shape)
    
    def forward(self, input):
        self.input = input
        self.output = np.zeros(self.output_shape)
        for k in range(self.input_depth):
            for i in range(self.output_shape[1]):
                for j in range(self.output_shape[2]):
                    subsquare = input[k,i*2:i*2+2,j*2:j*2+2]
                    for x in range(subsquare.shape[0]):
                        for y in range(subsquare.shape[1]):
                            self.input_max[k,i*2+x,j*2+y] = np.where(input[k,i*2+x,j*2+y] == np.amax(subsquare), 1, 0)
                    self.output[k,i,j] = np.amax(subsquare)
        #plt.imshow(self.output[1], cmap='Greys')
        #plt.show()
        return self.output

    def backwards(self, output_gradient, learning_rate):
        new_gradient = np.zeros(self.input_max.shape)
        output_gradient_depth, output_gradient_height, output_gradient_width = output_gradient.shape
        for k in range(output_gradient_depth):
            for i in range(output_gradient_width):
                for j in range(output_gradient_height):
                    subsquare = self.input_max[k,i*2:i*2+2,j*2:j*2+2]
                    for x in range(subsquare.shape[0]):
                        for y in range(subsquare.shape[1]):
                            new_gradient[k,i*2+x,j*2+y] = np.where(self.input_max[k,i*2+x,j*2+y] == 1, output_gradient[k,i,j], 0)
        return new_gradient