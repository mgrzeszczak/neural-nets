#!/usr/bin/env python3
import numpy as np


class NeuralNet:

    def __init__(self, dims):
        self.dims = dims
        self.weights = []
        self.biases = []
        for dim in zip(dims[1:], dims):
            self.weights.append(np.random.rand(*dim))
            self.biases.append(np.random.rand(dim[0], 1))

    def _train_test_split(self, x, y):
        train_size = int(0.8 * len(x))
        trainX = x[:train_size]
        trainY = y[:train_size]
        testX = x[train_size:]
        testY = y[train_size:]
        return trainX, trainY, testX, testY

    def feedforward(self, x):
        # activation for first layer
        a = [x]
        z = []
        # feed forward
        for l in range(len(self.dims)-1):
            z.append(self.weights[l].dot(a[l]) + self.biases[l])
            a.append(sigmoid(z[l]))
        return a

    def fit(self, x, y, max_iterations=10000):

        # trainX, trainY, testX, testY = self._train_test_split(x, y)

        for n in range(max_iterations):
            for _x, _y in zip(x, y):
                a = self.feedforward(_x)
                # output error
                d = [None] * (len(a))
                d[-1] = (a[-1] - _y)*sigmoid(a[-1], deriv=True)

                # backpropagation
                for i in range(len(a)-2, 0, -1):
                    w = self.weights[i]
                    d[i] = (w.T.dot(d[i+1]))*sigmoid(a[i], deriv=True)

                # gradient descent
                for i in range(len(self.biases)):
                    delta = d[i+1]
                    self.biases[i] -= delta
                    self.weights[i] -= np.outer(delta, a[i])
            predicted = self.predict(x)
            average_error = np.average(
                np.array(list(map(lambda k: self._error(k[0], k[1]), zip(predicted, y)))))

            if average_error < 0.001:
                print(f'trained after {n} iterations')
                break

    def _error(self, yhat, y):
        return 0.5 * np.linalg.norm(y-yhat)**2

    def predict(self, x):
        return np.array(list(map(lambda arg: self.feedforward(arg)[-1], x)))


def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))


def main():
    nn = NeuralNet([2, 3, 1])

    X = np.array([
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]]])
    Y = np.array(
        [[[0]],
         [[1]],
         [[1]],
         [[0]]])
    nn.fit(X, Y, max_iterations=10000)
    print(nn.predict(X))


if __name__ == '__main__':
    main()
