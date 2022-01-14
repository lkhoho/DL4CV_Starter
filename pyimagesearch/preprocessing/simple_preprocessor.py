import cv2
import numpy as np


class SimplePreprocessor(object):
    """
    A simple image preprocessor that resizes images to fixed size, ignoring the aspect ratio.
    """

    def __init__(self, width: int, height: int, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
