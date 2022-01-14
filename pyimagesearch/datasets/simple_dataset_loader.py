from typing import Any, List, Tuple
import numpy as np
import cv2
import os


class SimpleDatasetLoader(object):
    def __init__(self, preprocessors: List[Any]=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths: List[str], verbose=-1) -> Tuple[np.ndarray, np.ndarray]:
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.sep)[-2]

            for p in self.preprocessors:
                image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print('[INFO] processed {}/{}'.format(i + 1, len(imagePaths)))

        return np.array(data), np.array(labels)
