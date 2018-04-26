import numpy as np


class NeuralNetwork(object):
    def __init__(self, layers, alpha=0.1):
        """
        Initialize the list of weights matrices, then store the network architecture and learning rate.
        :param layers: A list of integers which represents the actual architecture of the feed-forward network. e.g.
        [2, 2, 1] would imply that our first input layer has two nodes, our hidden layer has two nodes, and our final
        output layer has one node.
        :param alpha: The learning rate of our neural network.
        """
        self.W = []  # a list of weights for each layer
        self.layers = layers
        self.alpha = alpha

        # start looping from the index of the first layer but stop before we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # the last two layers are a special case where the input connections need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network architecture
        return 'NeuralNetwork: {}'.format('-'.join(str(l) for l in self.layers))

    def sigmoid(self, x):
        """
        Activation function.
        :param x:
        :return:
        """
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        """
        Compute the derivative of the sigmoid function ASSUMING that 'x' has already been passed through the 'sigmoid'
        function.
        :param x:
        :return:
        """
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # Insert a column of 1's as the last entry in the feature matrix -- this little trick allows us to treat
        # the bias as a trainable parameter within the weight matrix.
        X = np.column_stack((X, np.ones(X.shape[0])))

        # loop over the desired number of epochs
        for epoch in np.arange(epochs):
            # loop over each individual data point and train our network on it
            for (x, target) in zip(X, y):
                self.fitPartial(x, target)

            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculateLoss(X, y)
                print('[INFO] epoch={}, loss={:.7f}'.format(epoch + 1, loss))

    def fitPartial(self, x, y):
        """
        :param x: Individual data point from our design matrix.
        :param y: The corresponding class label.
        :return:
        """
        # construct our list of output activations for each layer as our data point flows through the network; the
        # first activation is a special case -- it's just the input feature vector itself
        A = [np.atleast_2d(x)]

        # FEED-FORWARD:
        # loop over the layers in the network
        for layer in np.arange(len(self.W)):
            # feed-forward the activation at the current layer by taking the dot product between the activation and
            # the weight matrix -- this is called the 'net input' to the current layer
            net = A[layer].dot(self.W[layer])

            # compute the 'net output' is simply applying our nonlinear activation function to the net input
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of activations
            A.append(out)

        # BACK-PROPAGATION:
        # the first phase of back-propagation is to compute the difference between our *prediction* (the final output
        # activation in the activations list) and the true target value
        error = A[-1] - y

        # from here, we need to apply the chain rule and build our list of deltas 'D'; the first entry in the deltas is
        # simply the error of the output layer times the derivative of our activation function for the output value
        D = [error * self.sigmoid_deriv(A[-1])]

        # once you understand hte chain rule it becomes super easy to implement with a 'for' loop -- simply loop over
        # the layers in reverse order (ignoring the last two since we already have taken them into account)
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta of the *previous layer* dotted with the weight
            # matrix of the current layer, followed by multiplying the delta by the derivative of the nonlinear
            # activation function for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta *= self.sigmoid_deriv(A[layer])
            D.append(delta)

        # WEIGHT UPDATE:
        # reverse deltas
        D = D[::-1]

        # loop over the layers
        for layer in np.arange(len(self.W)):
            # update our weights by taking the dot product of the layer activations with their respective deltas, then
            # multiplying this value by some small learning rate and adding to our weight matrix -- this is where the
            # actual 'learning' takes place
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        # initialize the output prediction as the input features -- this value will be (forward) propagated through the
        # network to obtain the final prediction
        p = np.atleast_2d(X)

        if addBias:
            p = np.column_stack((p, np.ones(p.shape[0])))

        for layer in np.arange(len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculateLoss(self, X, target):
        targets = np.atleast_2d(target)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss
