# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries

import numpy as np
import pandas as pd
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Importing the dataset
dataset = pd.read_csv('D:/Git-Kaggle/Kaggle/digit-recognizer/train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the CNN!


# Initialising the CNN
classifier = Sequential()

# Trial 1
classifier.add(Convolution2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 24, nb_epoch = 60)
# Result in accuracy 0.95154761904761909

del classifier
del y_pred, y_pred1,y_test1
del cm
del i, row, N

# Trial 2
classifier = Sequential()
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu', input_dim = 784))
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 24, nb_epoch = 60)
# Result in accuracy 

del classifier
del y_pred, y_pred1,y_test1
del cm
del i, row, N

# Trial 3
classifier = Sequential()
classifier.add(Dense(output_dim = 350, init = 'uniform', activation = 'relu', input_dim = 784))
classifier.add(Dense(output_dim = 350, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 60)
# Result in accuracy 0.95773809523809528

del classifier
del y_pred, y_pred1,y_test1
del cm
del i, row, N

# Trial 4
classifier = Sequential()
classifier.add(Dense(output_dim = 350, init = 'uniform', activation = 'relu', input_dim = 784))
classifier.add(Dense(output_dim = 350, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 100)
# Result in accuracy 0.95392857142857146

del classifier
del y_pred, y_pred1,y_test1
del cm
del i, row, N

# Trial 5
classifier = Sequential()
classifier.add(Dense(output_dim = 250, init = 'uniform', activation = 'relu', input_dim = 784))
classifier.add(Dense(output_dim = 250, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 60)
# Result in accuracy 0.9553571428571429

del classifier
del y_pred, y_pred1,y_test1
del cm
del i, row, N

# Trial 6
classifier = Sequential()
classifier.add(Dense(output_dim = 250, init = 'uniform', activation = 'relu', input_dim = 784))
classifier.add(Dense(output_dim = 250, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# Result in accuracy 0.94119047619047624

del classifier
del y_pred, y_pred1,y_test1
del cm
del i, row, N

# Trial 7
classifier = Sequential()
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu', input_dim = 784))
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 500, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 100)
# Result in accuracy 0.40273809523809523???

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
N = y_pred.shape[0]
y_pred1 = np.zeros((1*N),int)
y_test1 = np.zeros((1*N),int)

for i,row in enumerate(y_pred):
    if row[0]:
        y_pred1[i] = 0
    elif row[1]:
        y_pred1[i] = 1
    elif row[2]:
        y_pred1[i] = 2
    elif row[3]:
        y_pred1[i] = 3
    elif row[4]:
        y_pred1[i] = 4
    elif row[5]:
        y_pred1[i] = 5
    elif row[6]:
        y_pred1[i] = 6
    elif row[7]:
        y_pred1[i] = 7
    elif row[8]:
        y_pred1[i] = 8
    elif row[9]:
        y_pred1[i] = 9

for i,row in enumerate(y_test):
    if row[0]:
        y_test1[i] = 0
    elif row[1]:
        y_test1[i] = 1
    elif row[2]:
        y_test1[i] = 2
    elif row[3]:
        y_test1[i] = 3
    elif row[4]:
        y_test1[i] = 4
    elif row[5]:
        y_test1[i] = 5
    elif row[6]:
        y_test1[i] = 6
    elif row[7]:
        y_test1[i] = 7
    elif row[8]:
        y_test1[i] = 8
    elif row[9]:
        y_test1[i] = 9

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred1)

AR = cm.diagonal().sum()/y_pred1.shape[0]
AR


testDS = pd.read_csv('D:/Git-Kaggle/Kaggle/digit-recognizer/test.csv')
testDS.head()
testDS.shape
Xtest = testDS.iloc[:, :].values
Xtest.shape
Xtest = sc.transform(Xtest)
y_final_pred = classifier.predict(Xtest)

N = y_final_pred.shape[0]

del submission1
del subtestDS
del row

submission1 = np.zeros((N,2),int)

for i,row in enumerate(y_final_pred):
    if row[0]:
        submission1[i,] = (i+1,0)
    elif row[1]:
        submission1[i,] = (i+1,1)
    elif row[2]:
        submission1[i,] = (i+1,2)
    elif row[3]:
        submission1[i,] = (i+1,3)
    elif row[4]:
        submission1[i,] = (i+1,4)
    elif row[5]:
        submission1[i,] = (i+1,5)
    elif row[6]:
        submission1[i,] = (i+1,6)
    elif row[7]:
        submission1[i,] = (i+1,7)
    elif row[8]:
        submission1[i,] = (i+1,8)
    elif row[9]:
        submission1[i,] = (i+1,9)
        
np.savetxt('D:/Git-Kaggle/Kaggle/digit-recognizer/submission1.csv',submission1, delimiter=',',header="ImageId,Label", comments='',fmt='%i')

subtestDS = pd.read_csv('D:/Git-Kaggle/Kaggle/digit-recognizer/submission1.csv')
subtestDS.head()
subtestDS.shape

sampleDS = pd.read_csv('D:/Git-Kaggle/Kaggle/digit-recognizer/sample_submission.csv')
sampleDS.head()
sampleDS.shape

