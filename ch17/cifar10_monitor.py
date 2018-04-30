import matplotlib
matplotlib.use("Agg")

import os
import argparse

from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from keras.datasets import cifar10

from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor


ap = argparse.ArgumentParser()
ap.add_argument('-b', '--useBN', type=int, default=1, help='use batch normalization or not')
ap.add_argument('-o', '--output', required=True, help='path to the output directory')
args = vars(ap.parse_args())
useBatchNormalization = True if args['useBN'] == 1 else False

# show information on the process ID
print('[INFO] process ID: {}'.format(os.getpid()))

# load CIFAR-10 dataset
print('[INFO] loading CIFAR-10 data...')
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
labelNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# initialize the SGD optimizer, but without any learning rate decay
print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10, useBatchNormalization=useBatchNormalization)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the set of callbacks
figPath = os.path.sep.join([args['output'], '{}.png'.format(os.getpid())])
jsonPath = os.path.sep.join([args['output'], '{}.json'.format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

# train the network
print('[INFO] training network...')
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=100, callbacks=callbacks, verbose=2)
