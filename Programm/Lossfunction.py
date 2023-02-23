import numpy as np
from Activationfunctions import Softmax

def cross(y_pred, y_true):
    y_pred = np.log(y_pred)
    return -1 * np.sum(y_true * np.where(y_pred == np.NINF, 0, y_pred))

def cross_prime(y_pred, y_true):
    return y_true/y_pred

pred = np.array(
[[1],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0]])

true = np.array(
[[0],
[0],
[0],
[0],
[1],
[0],
[0],
[0],
[0],
[0]])

#cross(pred,true)      
