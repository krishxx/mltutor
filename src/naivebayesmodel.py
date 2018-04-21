# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 06:49:44 2018

@author: Srikrishna.Sadula
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

print("Naive Bayes Model Example")

x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, Y)

#Predict Output 
predicted= model.predict([[1,2],[3,4]])

print (predicted)

score = accuracy_score(Y[1:2], predicted)