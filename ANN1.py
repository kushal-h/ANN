# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:46:30 2020

@author: KUSHAL H
"""
# Importing Librarries
import numpy as np
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
dataset1 = dataset

# Eliminating The Dummy Varibales

df = pd.get_dummies(dataset['Gender'], drop_first = True)
df1 = pd.get_dummies(dataset['Geography'], drop_first = True)

# Removing Unwanted data
dataset.drop('RowNumber', axis = 1, inplace = True )
dataset.drop('CustomerId', axis = 1, inplace = True )
dataset.drop('Surname', axis = 1, inplace = True )
dataset.drop('Geography', axis = 1, inplace = True )
dataset.drop('Gender', axis = 1, inplace = True )

# Concat
dataset = pd.concat([df1, dataset], axis = 1)
dataset = pd.concat([df, dataset], axis = 1)

# Taking x and y
x = dataset.iloc[:, :11].values
y = dataset.iloc[:, 11].values

# Spltting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# feature Scaling on the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Importing Keras Libraries

from keras.models import Sequential
from keras.layers import Dense

# Intilizing the ANN
model = Sequential()

# Adding the first input layer and hidden layer
model.add(Dense(units = 6, input_dim = 11, kernel_initializer = 'glorot_uniform', activation = 'relu'))

# Adding Second layer
model.add(Dense(units = 6, kernel_initializer='glorot_uniform', activation = 'relu'))

# Adding output layer
model.add(Dense(units = 1,  activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))   

#Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#Fitting ANN with Training sets
model.fit(x_train, y_train, batch_size = 30, epochs = 10)

# Predicting
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

'''
Single prediction details
Geography : France
Credit Score : 600
Gender : Male
Age : 40
Tennure : 3
Balance : 60,000
No of Products : 2
Has Credit Card : Yes
Is Active Member : Yes
Estimated Salary is : 50,000
Predict Whether the Customer will leave the bank or not
'''


new_pred = model.predict(sc.transform(np.array([[1, 0, 0, 600, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)


