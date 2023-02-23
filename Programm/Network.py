import numpy as np

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):

    for e in range(epochs):
        error = 0
        for x, y in zip(x_train[:10], y_train[:10]):
            y = np.array([y]).T
            # forward
            output = predict(network, x)
            print("Ausgabe",output)
            print("Richtig",y)
            # error
            error += loss(output, y)
            print("Loss-Wert",loss(output, y))

            # backward
            grad = loss_prime(output, y)
            print("Berechneter Gradient",grad)
            for layer in reversed(network):
                grad = layer.backwards(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")