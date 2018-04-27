from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class MiniVGGNet(object):
    @staticmethod
    def build(width, height, depth, classes, useBatchNormalization=True):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1  # the index of channel, -1 implies that the channel is at the last dimension

        # switch to 'channel_first' if necessary
        if K.image_data_format() == 'channel_first':
            inputShape = (depth, height, width)
            chanDim = 1

        # first CONV => RELU => BN => CONV => RELU => BN => POOL => DO layer set
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        if useBatchNormalization:
            model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        if useBatchNormalization:
            model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => BN => CONV => RELU => BN => POOL => DO layer set
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        if useBatchNormalization:
            model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        if useBatchNormalization:
            model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU => BN => DO layers
        model.add(Flatten())
        model.add(Dense(units=512))
        model.add(Activation('relu'))
        if useBatchNormalization:
            model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))  # typically p=0.5 is used in between FC layers

        # final softmax classifier
        model.add(Dense(units=classes))
        model.add(Activation('softmax'))

        # return the constructed model
        return model

