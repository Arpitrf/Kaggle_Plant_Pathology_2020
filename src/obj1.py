#import general libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#keras imports
# Importing the Keras libraries and packages
from keras import regularizers
from keras.models import Sequential, model_from_json, Model
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop, Adagrad, Adadelta, Adam, SGD
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping


class cnn_architecture():
    def __init__(self, learn_rate, mode='binary', output_neurons=1):
        self.h, self.w = 224, 224 #image height, width
        self.mode = mode
        self.batch, self.epoc = 1, 1
        self.lr = learn_rate
        self.last = output_neurons

    def create_model(self):
        # SET ALL THE PARAMETERS
        # LOAD VGG16
        input_tensor = Input(shape=(self.h,self.w,3))
        model = applications.VGG16(weights='imagenet', 
                                   include_top=False,
                                   input_tensor=input_tensor)
       
        # CREATE AN "REAL" MODEL FROM VGG16 BY COPYING ALL THE LAYERS OF VGG16
        new_model = Sequential()
        for l in model.layers:
            new_model.add(l)
        
        # CREATE A TOP MODEL
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1000, activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(18, activation='relu'))
        top_model.add(Dense(self.last, activation='sigmoid'))
        
        # CONCATENATE THE TWO MODELS
        new_model.add(top_model)
        
        # LOCK THE TOP CONV LAYERS        
        for layer in new_model.layers[:15]:
            layer.trainable = False
        
        return new_model

    #rescale, split in train/ validation set, create batches
    def image_gen(self, train, test=None, train_dir=None, test_dir=None,mode='binary'):
        labels = train.columns.values

        print('number of labels/ columns for y_col: ',len(labels))
        print('begin data pre-processing')

        train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range=30,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           zoom_range=0.5,
                                           validation_split = 0.2)

        test_datagen = ImageDataGenerator(rescale = 1./255.)

        train_gen = train_datagen.flow_from_dataframe(dataframe = train,
                                                 directory = train_dir,
                                                 x_col = 'path',
                                                 y_col = labels[3:],
                                                 subset = "training",
                                                 class_mode = self.mode,
                                                 target_size = (self.h,self.w),
                                                 batch_size = self.batch,
                                                 shuffle = True)

        val_gen = train_datagen.flow_from_dataframe(dataframe = train,
                                                 directory = train_dir,
                                                 x_col = 'path',
                                                 y_col = labels[3:],
                                                 subset = "validation",
                                                 class_mode = self.mode,
                                                 target_size = (self.h,self.w),
                                                 batch_size = self.batch,
                                                 shuffle = True)

        print('ended data pre-processing')
        #return train_gen, val_gen, test_gen
        return train_gen, val_gen
if __name__ == '__main__':
	path = '../data/'
	df = pd.read_csv(path + 'train.csv')
	test_data = pd.read_csv(path + 'test.csv')

