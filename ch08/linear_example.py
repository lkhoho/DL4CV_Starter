import numpy as np
import cv2


np.random.seed(1)
imageSize = (32, 32, 3)
labels = ['dog', 'cat', 'panda']  # initialize initial labels
D = imageSize[0] * imageSize[1] * imageSize[2]  # feature dimension
K = len(labels)  # number of labels

# randomly initialize the weight matrix and bias vector -- in a *real* training and classification task, these
# parameters would be *learned* by our model, but for the sake of this example, let's use random values.
W = np.random.randn(K, D)
b = np.random.randn(K)

# load our example image, resize it, and then flatten it into our "feature vector" representation
orig = cv2.imread('dog.png')
image = cv2.resize(orig, (imageSize[0], imageSize[1])).flatten()

# compute the output scores by taking the dot product between the weight matrix and image pixels, followed by adding in
# the bias
scores = W.dot(image) + b

# loop over the scores + labels and display them
for (label, score) in zip(labels, scores):
    print('[INFO] {}: {:.2f}'.format(label, score))

# draw the label with the highest score on the image as our prediction
cv2.putText(orig, 'Label: {}'.format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# display our input image
cv2.imshow('Image', orig)
cv2.waitKey(0)
