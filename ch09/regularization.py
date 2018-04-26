import argparse
from imutils import paths

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from pyimagesearch.datasets.simple_dataset_loader import SimpleDatasetLoader
from pyimagesearch.preprocessing.simple_preprocessor import SimplePreprocessor


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to input dataset')
ap.add_argument('-j', '--jobs', type=int, default=-1, help='# of jobs for k-NN distance (-1 uses all available cores)')
args = vars(ap.parse_args())

print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 32 * 32 * 3))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

# apply different types of regularization terms
for r in (None, 'l1', 'l2'):
    # train the SGD classifier using a softmax loss function and the specified regularization function for 10 epochs
    print('[INFO] training model with \'{}\' penalty'.format(r))
    model = SGDClassifier(loss='log', penalty=r, max_iter=10, learning_rate='constant', eta0=0.01, random_state=42)
    model.fit(trainX, trainY)

    # evaluate the classifier
    acc = model.score(testX, testY)
    print('[INFO] \'{}\' penalty accuracy: {:.2f}%'.format(r, acc * 100))

