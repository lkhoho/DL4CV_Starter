import argparse

from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10

from pyimagesearch.nn.conv.minivggnet import MiniVGGNet


ap = argparse.ArgumentParser()
ap.add_argument('-w', '--weights', required=True, help='path to the best model weights file')
args = vars(ap.parse_args())

print('[INFO] loading CIFAR-10 data...')
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype('float') / 255.0
testX = testX.astype('float') / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print('[INFO] initialize the optimizer and model...')
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# construct the callback to save only the *best* model to disk based on the validation loss
checkpoint = ModelCheckpoint(args['weights'], monitor='val_loss', save_best_only=True, verbose=1)
callbacks = [checkpoint]

print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=2)
