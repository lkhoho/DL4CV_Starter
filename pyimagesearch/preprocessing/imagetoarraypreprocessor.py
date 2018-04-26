from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor(object):
    """
    Class that wraps the Keras utility function img_to_array() that correctly
    rearranges the dimension of the image
    """

    def __init__(self, dataFormat=None):
        """
        :param dataFormat: Defaults to None to use setting in keras.json.
        Optionally, 'channels_first' or 'channels_last' can be passed.
        """
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the Keras utility function that correctly rearranges the
        # dimension of the image
        return img_to_array(image, data_format=self.dataFormat)
