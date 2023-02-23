import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils

from DenseLayer import DenseLayer
from Convolutional import Convolutional
from Lossfunction import cross, cross_prime
from Flattening import Flattening
from Activationfunctions import ReLU, Softmax
from MaxPooling import MaxPooling
from Network import train, predict

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255
x_train = x_train.reshape(len(x_train),1,28,28)
x_test = x_test.reshape(len(x_test),1,28,28)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#plt.imshow(x_train[0,0], cmap='Greys')
#plt.show()

#TODO Underflow und Overflow fixxen, sodass komplexere Architekturen gebaut werden können
#TODO Sandbox benutzen und rumprobieren. Worst Szenario: Alles selber ausrechnen

network = [
    Convolutional((1,28,28),3,3),
    ReLU(),
    MaxPooling((3,26,26)),
    Convolutional((3,13,13),3,9),
    ReLU(),
    MaxPooling((9,11,11)),
    Flattening((9,6,6), (9*6*6,1)),
    DenseLayer(9*6*6,10),
    Softmax()
]

train(
    network,
    cross,
    cross_prime,
    x_train,
    y_train,
    epochs=1,
    learning_rate=0.1
)

'''
for x, y in zip(x_test, y_test):

    output = predict(network, x)
    counter = 0
    if(np.argmax(output) == np.argmax(y)):
        counter += 1
print("Richtig:", counter)
'''
