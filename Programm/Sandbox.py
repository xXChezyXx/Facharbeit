import numpy as np
from DenseLayer import DenseLayer
from Convolutional import Convolutional
from Lossfunction import cross, cross_prime
from Flattening import Flattening
from Activationfunctions import ReLU, Softmax
from MaxPooling import MaxPooling
from Network import train, predict

network = [
    Convolutional((1,28,28),5,5),
    ReLU(),
    MaxPooling((5,24,24)),
    Flattening((5,12,12), (5*12*12,1)),
    DenseLayer(5*12*12,10),
    Softmax()
]

true = np.array([[0,0,0,0,0,0,0,0,0,1]]).T

matrix = np.array([np.random.uniform(0, 1, (6, 6)) for i in range(3)])

conv = Convolutional((1,28,28),3,5)
out = conv.forward(matrix)
print("Conv")
print(out)
print("")
relu = ReLU()
out = relu.forward(out)
print("Relu")
print(out)
print("")
pool = MaxPooling((5,26,26))
out = pool.forward(out)
print("Pooling")
print(out)

conv1 = Convolutional((5,13,13),3,25)
out = conv1.forward(out)
print("Conv")
print(out)
print("")
relu1 = ReLU()
out = relu1.forward(out)
print("Relu")
print(out)
print("")
pool1 = MaxPooling((25,11,11))
out = pool1.forward(out)
print("Pooling")
print(out)

flat = Flattening((25,6,6),(25*6*6,1))
out = flat.forward(out)
print("Flat")
print(out)
dense = DenseLayer(25*6*6,100)
out = dense.forward(out)
print("Dense")
print(out)
relu2 = ReLU()
out = relu2.forward(out)
print("Relu")
print(out)
print("")
dense1 = DenseLayer(100,10)
out = dense1.forward(out)
print("Dense")
print(out)
s = Softmax()
out = s.forward(out)
print("Softmax")
print(out)
print(cross(out,true))
grad = cross_prime(out,true)
print(cross_prime(out,true))
print(s.backwards(grad, 0.2))