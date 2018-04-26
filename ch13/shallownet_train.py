from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pylab as plt
import numpy as np
import argparse

from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simple_preprocessor import SimplePreprocessor
from pyimagesearch.datasets.simple_dataset_loader import SimpleDatasetLoader
from pyimagesearch.nn.conv.shallownet import ShallowNet


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print('[INFO] compiling model...')
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=2)

print('[INFO] serializing network...')
model.save(args['model'])

print('[INFO] evaluate network...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=['cat', 'dog', 'pandas']))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='validation_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='train_accuracy')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='validation_accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
