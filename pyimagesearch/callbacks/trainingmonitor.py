import os
import simplejson as json
import numpy as np
import matplotlib.pylab as plt
from keras.callbacks import BaseLogger


class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        """
        Construct a TrainingMonitor instance.
        :param figPath: The path to the output plot that we can use to visualize loss and accuracy over time.
        :param jsonPath: The path used to serialize the loss and accuracy values as a JSON file. This is useful if you
        want to use the training history to create custom plots of your own.
        :param startAt: The starting epoch that training is resumed at when using "ctrl + c training".
        """
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs=None):
        # initialize the history of loss
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and trim any entries that are past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs=None):
        # loop over the logs and update the loss, accuracy, etc. for the entire training process
        for (k, v) in logs.items():
            log = self.H.get(k, [])
            log.append(v)
            self.H[k] = log

        # check to see if the training history should be serialized to file
        if self.jsonPath is not None:
            with open(self.jsonPath, 'w') as fp:
                fp.write(json.dumps(self.H))

        # ensure at least two epochs have passed before plotting (epoch starts at zero)
        if len(self.H['loss']) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H['loss']))
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(N, self.H['loss'], label='train_loss')
            plt.plot(N, self.H['val_loss'], label='validation_loss')
            plt.plot(N, self.H['acc'], label='train_accuracy')
            plt.plot(N, self.H['val_acc'], label='validation_accuracy')
            plt.xlabel('Epoch #')
            plt.ylabel('Loss/Accuracy')
            plt.title('Training Loss and Accuracy [Epoch {}]'.format(len(self.H['loss'])))
            plt.legend()

            # save the figure
            plt.savefig(self.figPath)
            plt.close()
