from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pyimagesearch.nn.neuralnetwork import NeuralNetwork

print('[INFO] loading MNIST (sample) dataset...')
digits = datasets.load_digits()
data = digits.data.astype('float')
data = (data - data.min()) / (data.max() - data.min())
print('[INFO] samples: {}, dim: {}'.format(data.shape[0], data.shape[1]))

(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print('[INFO] training network...')
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print('[INFO] {}'.format(nn))
nn.fit(trainX, trainY, epochs=1000)

print('[INFO] evaluating network...')
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))
