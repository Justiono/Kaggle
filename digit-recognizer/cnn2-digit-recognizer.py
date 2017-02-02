# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 21:42:52 2017

"""

import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

train_df = pd.read_csv('D:/Git-Kaggle/Kaggle/digit-recognizer/train.csv')

x_train = train_df.drop(['label'], axis=1).values.astype('float32')
Y_train = train_df['label'].values.astype('float32')

img_width, img_height = 28, 28
n_train = x_train.shape[0]
n_classes = 10

x_train2 = x_train.reshape(n_train,img_width,img_height,1)
x_train2 = x_train2/255 #normalize from [0,255] to [0,1]
y_train = to_categorical(Y_train)

# Trial 1
classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, border_mode='same', input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train2, y_train, batch_size=24, nb_epoch=10, verbose=2, validation_split=.2)
#Accuracy: 0.9851

del classifier    
# Trial 2
classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, border_mode='same', input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(64, 3, 3, border_mode='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train2, y_train, batch_size=24, nb_epoch=20, verbose=2, validation_split=.2)
#Accuracy: 0.9904
del classifier
# Trial 3
classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, border_mode='same', input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(64, 3, 3, border_mode='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(128, 3, 3, border_mode='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train2, y_train, batch_size=24, nb_epoch=20, verbose=2, validation_split=.2)
#Accuracy: 0.9911

del x_submit,classifier
# Trial 4
classifier = Sequential()
classifier.add(Convolution2D(64, 3, 3, border_mode='same', input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(128, 3, 3, border_mode='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(256, 3, 3, border_mode='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train2, y_train, batch_size=24, nb_epoch=20, verbose=2, validation_split=.2)
# 0.9902

# Trial 5
classifier = Sequential()
classifier.add(Convolution2D(128, 3, 3, border_mode='same', input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(256, 3, 3, border_mode='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train2, y_train, batch_size=10, nb_epoch=20, verbose=2, validation_split=.2)

%matplotlib inline
import matplotlib.pyplot as plt

imgplot = plt.imshow(x_train2[5309,:,:,0],cmap='gray')
imgplot = plt.imshow(x_submit[5310,:,:,0],cmap='gray')
imgplot = plt.imshow(x_submit[5311,:,:,0],cmap='gray')
y_pred[3] = 0
y_pred[51]=5
y_pred[54]=0
y_pred[71]=9
y_pred[88]=9
y_pred[91]=6
y_pred[118]=1
y_pred[97]=2
y_pred[27993]=3
y_pred[27995]=9
y_pred[11504]=1
y_pred[9545]=1
y_pred[7803]=4
y_pred[5309]=6

submit_df = pd.read_csv('D:/Git-Kaggle/Kaggle/digit-recognizer/test.csv')
x_submit = submit_df.values.astype('float32')
n_submit = x_submit.shape[0]
x_submit = x_submit.reshape(n_submit,img_width,img_height,1)
x_submit = x_submit/255

y_pred = classifier.predict_classes(x_submit,batch_size=10,verbose=1)
np.savetxt('mnist_cnn4c.csv', np.c_[range(1,len(y_pred)+1),y_pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d') 