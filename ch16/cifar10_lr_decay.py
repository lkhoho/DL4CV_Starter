import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pylab as plt
import numpy as np
import argparse

from pyimagesearch.nn.conv.minivggnet import MiniVGGNet


ap = argparse.ArgumentParser()
ap.add_argument('-b', '--useBN', type=int, default=1, help='use batch normalization or not')
ap.add_argument('-f', '--factor', type=float, default=0.25, help='drop factor for learning rate')
ap.add_argument('-m', '--model', required=True, help='path to the model trained')
ap.add_argument('-o', '--output', required=True, help='path to the output loss/accuracy plot')
args = vars(ap.parse_args())
useBatchNormalization = True if args['useBN'] == 1 else False


def stepDecay(epoch):
    # initialize the base initial learning rate, drop factor, and epochs to drop every
    initAlpha = 0.01
    factor = args['factor']
    dropEvery = 5  # drop the learning rate for every 5 epochs

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    # return the learning rate
    return float(alpha)


print('[INFO] loading CIFAR-10 data...')
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
labelNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# define the set of callbacks to be passed to the model during training
callbacks = [LearningRateScheduler(stepDecay)]

print('[INFO] compiling model...')
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10, useBatchNormalization=useBatchNormalization)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=2)

print('[INFO] serializing network...')
model.save(args['model'])

print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 40), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 40), H.history['val_loss'], label='validation_loss')
plt.plot(np.arange(0, 40), H.history['acc'], label='train_accuracy')
plt.plot(np.arange(0, 40), H.history['val_acc'], label='validation_accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.title('CIFAR-10 Learning Rate Decay (factor={})'.format(str(args['factor'])))
plt.legend()
plt.savefig(args['output'])
