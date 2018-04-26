import cv2


class SimplePreprocessor(object):
    """
    A simple image preprocessor that resizes images to fixed size, ignoring the aspect ratio.
    """

    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
