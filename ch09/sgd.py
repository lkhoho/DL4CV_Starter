# stochastic gradient descent

import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pylab as plt

def sigmoidActivation(x):
    """
    Compute the sigmoid activation value for a given input.
    :param x: The given input.
    :return:
    """

    return 1.0 / (1 + np.exp(-x))


def sigmoidDeriv(x):
    """
    Compute the derivative of the sigmoid function ASSUMING that the input 'x' has already been passed through the
    sigmoid activation function.
    :param x: The input that has already been passed through the sigmoid activation function.
    :return:
    """

    return x * (1 - x)


def predict(X, W):
    """
    A simple prediction function that applies sigmoid activation function and thresholds it based on whether the neuron
    is firing(1) or not (0).
    :param X: The input data points X.
    :param W: The weights matrix.
    :return: The predicted labels.
    """

    preds = sigmoidActivation(X.dot(W))
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1

    return preds


def nextBatch(X, y, batchSize):
    """
    Loop over our dataset 'X' in mini-batches, yielding a tuple of the current batched data and labels.
    :param X: The input dataset.
    :param y: The input labels.
    :param batchSize: The mini-batch size.
    :return:
    """

    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], y[i:i + batchSize])


ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', type=int, default=100, help='# of epochs')
ap.add_argument('-a', '--alpha', type=float, default=0.01, help='learning rate')
ap.add_argument('-b', '--batch_size', type=int, default=32, help='size of SGD batches')
args = vars(ap.parse_args())

# generate a 2-class classification problem with 1000 data points, where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# add bias terms
X = np.column_stack((X, np.ones(X.shape[0])))
# X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

# initialize weight matrix and list of losses
print('[INFO] training...')
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over the desired number of epochs
for epoch in np.arange(0, args['epochs']):
    epochLoss = []  # total loss for this epoch

    # loop over our data in batches
    for (batchX, batchY) in nextBatch(X, y, args['batch_size']):
        # take the dot product between our current batch of features and the weight matrix, then pass this value
        # through our activation function
        preds = sigmoidActivation(batchX.dot(W))

        # now that we have our predictions, we need to determine the 'error', which is the difference between our
        # predictions and the true values
        error = preds - batchY
        epochLoss.append(np.sum(error ** 2))

        # the gradient descent update is the dot product between our 1) current batch and 2) the error of the
        # sigmoid derivative of our predictions
        d = error * sigmoidDeriv(preds)
        gradient = batchX.T.dot(d)

        # in the update stage, all we need to do is 'nudge' the weight matrix in the negative direction of the gradient
        # (hence the term 'gradient descent' by taking a small step towards a set of 'more optimal' parameters
        W += -args['alpha'] * gradient

    # update our loss history by taking the average loss across all batches
    loss = np.average(epochLoss)
    losses.append(loss)

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print('[INFO] epoch={}, loss={:.7f}'.format(epoch + 1, loss))

# evaluate our model
print('[INFO] evaluating...')
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY[:, 0], s=30)

# construct a figure that plots the loss over time
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, args['epochs']), losses)
plt.title('Training Loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()

