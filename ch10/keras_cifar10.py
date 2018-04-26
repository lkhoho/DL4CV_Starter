import numpy as np
import argparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pylab as plt


ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to the output loss/accuracy plot')
args = vars(ap.parse_args())

# load the training and testing data, scale it into the range [0.0, 1.0], then reshape the design matrix
# the CIFAR-10 author has decided 50000 images for training and 10000 images for testing
print('[INFO] loading CIFAR-10 data...')
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0
trainX = trainX.reshape((trainX.shape[0], 3072))  # each image is represented by 32 * 32 * 3
testX = testX.reshape((testX.shape[0], 3072))

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# define the 3072-1024-512-10 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

# train the model using SGD
print('[INFO] training network...')
sgd = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

# evaluate the network
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=labelNames))

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='validation_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='train_accuracy')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='validation_accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss or Accuracy')
plt.legend()
plt.savefig(args['output'])
