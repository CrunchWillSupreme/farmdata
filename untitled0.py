# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:05:12 2019

@author: willh
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
import datetime as dt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score

data = pd.read_csv(r'C:\Users\willh\farmstate\exercise_04_train.csv')
with open(r'C:\Users\willh\farmstate\instructions.txt') as f:
    instructions = f.read()

data.head()
data.describe()
data.shape
data.dtypes

data.isnull().sum()
data.y.value_counts()

total = len(data)
yes = data['y'][data['y'] == 1].count()
yes_percent = yes/total *100

notnull = data.dropna()
notnull.shape
notnull.isnull().sum()
corr = data.corr()
ax = sns.heatmap(corr)

sns.scatterplot(data['x0'],data['x21'])


fig = plt.figure(figsize = (20, 25))
j = 0
for i in notnull.columns[24:48]:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(notnull[i][notnull['y']==0], color='r', label = 'No')
    sns.distplot(notnull[i][notnull['y']==1], color='g', label = 'Yes')
    plt.legend(loc='best')
fig.suptitle('Data')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()

y = data.y
X = data.drop(columns = ['y'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
