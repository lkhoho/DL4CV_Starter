from imutils import paths
import argparse

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pyimagesearch.preprocessing.simple_preprocessor import SimplePreprocessor
from pyimagesearch.datasets.simple_dataset_loader import SimpleDatasetLoader


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to input dataset')
ap.add_argument('-k', '--neighbors', type=int, default=1, help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', type=int, default=-1, help='# of jobs for k-NN distance (-1 uses all available cores)')
args = vars(ap.parse_args())

print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

imageDim = (32, 32)
nColorChannels = 3
sp = SimplePreprocessor(*imageDim)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=1000)
data = data.reshape(data.shape[0], imageDim[0] * imageDim[1] * nColorChannels)
print('[INFO] features matrix: {:.1f} MB'.format(data.nbytes / (1024 * 1000.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

print('[INFO] evaluating k-NN classifiers...')
model = KNeighborsClassifier(n_neighbors=args['neighbors'], n_jobs=args['jobs'])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))

