# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder


myDS = pd.read_csv('Data.csv')
myDS.info()
myDS.describe()


X = myDS.iloc[:,:-1].values
Y = myDS.iloc[:,-1:].values

# Xdf = pd.DataFrame(myds.iloc[:,:-1].values)
# Ydf = pd.DataFrame(myds.iloc[:,-1:].values)

imputer = Imputer(missing_values = 'NaN', strategy='mean' , axis = 0)
imputer = imputer.fit(X[:,1:])

X[:,1:] = imputer.transform(X[:,1:])

# imputerDF = Imputer(missing_values = 'NaN', strategy='mean' , axis = 0)
# imputerDF = imputerDF.fit(Xdf[:,1:])
# Xdf[:,1:] = imputerDF.transform(Xdf[:,1:])

labelencoderX = LabelEncoder()
X[:,0] = labelencoderX.fit_transform(X[:,0])
onehot = OneHotEncoder(categorical_features = [0])
X=onehot.fit_transform(X).toarray()

labelencoderY = LabelEncoder()
Y = labelencoderY.fit_transform(Y)