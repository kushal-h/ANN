# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 08:57:43 2020

@author: KUSHAL H
"""
# importing numpy and pandas
import numpy as np
import pandas as pd

# reading csv 

dataset = pd.read_csv('train.csv')
dataset1=dataset

# filling NaN in gender with No gender
dataset["Gender"].fillna("No Gender", inplace = True)  

# droping NAN values
new_data = dataset.dropna(axis = 0, how ='any')  

# replacing wrong values

new_data.replace(to_replace = '3+', value = 4,inplace = True, limit=None, regex=True,  method='pad') 
# Eliminating The Dummy Varibales

dm1 = pd.get_dummies(new_data['Gender'], drop_first = True)
dm2 = pd.get_dummies(new_data['Married'], drop_first = True)
dm3 = pd.get_dummies(new_data['Education'], drop_first = True)
dm4 = pd.get_dummies(new_data['Self_Employed'], drop_first = True)
dm5 = pd.get_dummies(new_data['Property_Area'], drop_first = True)

# removing unwanted datas
new_data.drop('Loan_ID', axis = 1, inplace = True)
new_data.drop('Gender', axis = 1, inplace = True)
new_data.drop('Married', axis = 1, inplace = True)
new_data.drop('Education', axis = 1, inplace = True)
new_data.drop('Self_Employed', axis = 1, inplace = True)
new_data.drop('Property_Area', axis = 1, inplace = True)

# concat
new_data = pd.concat([dm1, new_data], axis = 1)
new_data = pd.concat([dm2, new_data], axis = 1)
new_data = pd.concat([dm3, new_data], axis = 1)
new_data = pd.concat([dm4, new_data], axis = 1)
new_data = pd.concat([dm5, new_data], axis = 1)

# sepearting X and Y
x = new_data.iloc[:, :12].values
y = new_data.iloc[:, 12].values

# Spltting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# feature Scaling on the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Importing Keras Libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Intilizing the ANN
model = Sequential()

# Adding the first input layer and hidden layer
model.add(Dense(units = 6, input_dim = 12, kernel_initializer = 'glorot_uniform', activation = 'relu'))
model.add(Dropout(rate = 0.1))

# Adding Second layer
model.add(Dense(units = 6, kernel_initializer='glorot_uniform', activation = 'relu'))
model.add(Dropout(rate = 0.1))

# Adding third layer
model.add(Dense(units = 6, kernel_initializer='glorot_uniform', activation = 'relu'))
model.add(Dropout(rate = 0.1))

# Adding output layer
model.add(Dense(units = 1,  activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))  
 
#Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#Fitting ANN with Training sets
model.fit(x_train, y_train, batch_size = 30, epochs = 100)