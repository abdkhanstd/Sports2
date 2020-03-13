"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Activation,MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
from keras import regularizers
from keras.layers.normalization import BatchNormalization
import functools
import keras.metrics
from capsuels import Capsule

from keras.models import Sequential, load_model
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import time
from keras.optimizers import SGD



import sys

class ResearchModels():


    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=1536):

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
        top3_acc.__name__ = 'top3_acc'
        
        top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
        top5_acc.__name__ = 'top5_acc'


        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy',top3_acc,top5_acc]
        # Get the appropriate model.
        if self.saved_model is not None:
            top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
            top3_acc.__name__ = 'top3_acc'
        
            top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
            top5_acc.__name__ = 'top5_acc'

            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model,custom_objects={'top3_acc': top3_acc,'top5_acc': top5_acc,'Capsule':Capsule})
        elif model == 'capsule':
            print("Loading capsule network.")
            self.input_shape = (seq_length, features_length)
            self.model = self.capsule()            
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        
        print(self.model.summary())
        
 

            
    def capsule(self):
    
        input_image = Input(shape=(self.input_shape))
        A1 = Capsule(32, 32, 3, True)(input_image)
        A2 = Capsule(14, 32, 3, True)(A1)
        A3 = Capsule(14, 32, 3, True)(A2)
        cat=Concatenate(axis=-1)([A2,A3])
        output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(cat)
        model = Model(inputs=input_image, outputs=output)
        

        return model
                
     

