import numpy as np


class Perceptron(object):
    def __init__(self, N, alpha=0.1):
        """
        Constructs a instance of Perceptron instance.
        :param N: The number of columns in input feature vector.
        :param alpha: The learning rate.
        """

        # initialize the weight matrix and store the learning rate
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        """
        Serves as activation function.
        :param x:
        :return:
        """

        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        """
        Fit a model to the data.
        :param X: The actual training data.
        :param y: The target output class labels.
        :param epochs: The number of epochs the Perceptron will train for.
        :return:
        """

        X = np.column_stack((X, np.ones((X.shape[0]))))

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                p = self.step(np.dot(x, self.W))

                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        X = np.atleast_2d(X)

        if addBias:
            X = np.column_stack((X, np.ones((X.shape[0]))))

        return self.step(np.dot(X, self.W))
