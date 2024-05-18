# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:14:35 2024

@author: Yamaç
"""

from ucimlrepo import fetch_ucirepo 
import numpy as np
import tensorflow as tf  

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features
y = wine_quality.data.targets
  
for i in range(len(y)):
    if y.iloc[i, 0] <=4:
        y.iloc[i, 0] = 0
    elif y.iloc[i, 0] <= 7:
        y.iloc[i, 0] = 1
    elif y.iloc[i, 0] <= 9:
        y.iloc[i, 0] = 2

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
y = np.array(ct.fit_transform(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=32, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=3, activation='softmax'))

ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy', 'Precision', 'F1Score'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import accuracy_score, precision_score

ac = accuracy_score(y_test, y_pred)
print(ac)
pc = precision_score(y_test, y_pred, average='micro')
print(pc)