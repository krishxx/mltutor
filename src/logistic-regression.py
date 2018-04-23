# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:37:50 2018

@author: Srikrishna.Sadula
"""

# banking services data for logistic regression

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


#preparing data
data = pd.read_csv("..//data//banking.csv", header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))

data.head()
#The education column of the dataset has many categories and we need to reduce the categories for a better modelling
data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

#data exploration
data['y'].value_counts()
sns.countplot(x='y', data=data, palette='hls')
plt.show()
plt.savefig('count_plot')
data.groupby('y').mean()
data.groupby('job').mean()
#.....


pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')