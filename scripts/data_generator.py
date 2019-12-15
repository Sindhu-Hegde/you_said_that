import numpy as np
import keras
from numpy import newaxis

class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, keys, audio, identity, gt, batch_size=64, shuffle=True):

        'Initialization'
        self.batch_size = batch_size
        self.keys = keys
        self.audio = audio
        self.identity = identity
        self.gt = gt
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.keys) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_keys = [self.keys[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_keys)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.keys))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_keys):
        'Generates data containing batch_size samples' 
        
        X_audio = []
        X_identity = []
        y = np.empty((self.batch_size, 112, 112, 3))

        # Generate data
        for i, key in enumerate(batch_keys):
            # Store data
            audio_data = self.audio[key]

            fname = key.split(".")[0] + '.jpg'  
            identity_data = self.identity[fname]
            gt_data = self.gt[fname]

            x_audio = np.array(audio_data)
            x_audio = x_audio.astype('float32') 
            x_audio = x_audio[:, :, newaxis]

            x_identity = np.array(identity_data)
            x_identity = x_identity.astype('float32') / 255.

            y_train = np.array(gt_data)
            y_train = y_train.astype('float32') / 255.

            X_audio.append(x_audio)
            X_identity.append(x_identity)
            
            # Store gt
            y[i,] = y_train

        X = [np.array(X_audio), np.array(X_identity)]

        return X, y