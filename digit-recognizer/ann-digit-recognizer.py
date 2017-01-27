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

# Part 2 - Now let's make the ANN!


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 397, init = 'uniform', activation = 'relu', input_dim = 784))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 397, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 50, nb_epoch = 60)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

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

