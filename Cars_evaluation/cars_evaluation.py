# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:47:04 2018

@author: Ravikiran.Tamiri
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('carsdata.csv')

# Assign names to Columns
dataset.columns = ['buying','maint','doors','persons','lug_boot','safety','classes']
# Encode Data
dataset.buying.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)
dataset.maint.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)
dataset.doors.replace(('2','3','4','5more'),(1,2,3,4), inplace=True)
dataset.persons.replace(('2','4','more'),(1,2,3), inplace=True)
dataset.lug_boot.replace(('small','med','big'),(1,2,3), inplace=True)
dataset.safety.replace(('low','med','high'),(1,2,3), inplace=True)
dataset.classes.replace(('unacc','acc','good','vgood'),(1,2,3,4), inplace=True)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#plot matlab 
plt.hist(dataset.classes)
dataset.hist()

#classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =7,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#y_pred_train = classifier.predict(X_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#apply k-fold cross validation
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator=classifier,X = X_train,y = y_train,cv = 10)
accuracies.mean()
accuracies.std()

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
#score_train = accuracy_score(y_train,y_pred_train)


