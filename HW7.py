# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:39:39 2018

@author: hongy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score



print("My name is Hongyu Zhang")
print("My NetID is: hongyuz2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

#read the file
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', \
                   'Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols', \
                   'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
cols=['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', \
                   'Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols', \
                   'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'Hue']].values


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1, 
                                                    random_state=42)
#standard 
sc = StandardScaler() 
sc.fit(X_train) 
X_train_std = sc.transform(X_train) 
X_test_std = sc.transform(X_test)
cv_train=[]    
accuracy_test=[]
for est in range(1,31):
    print("n_estimators = {}".format(est))
    rfc = RandomForestClassifier(n_estimators=est,random_state=21)
    # Fit rf to the training set    
    rfc.fit(X_train_std,y_train) 
    # Predict the test set labels
    y_pred = rfc.predict(X_test_std) 
    k_fold = KFold(len(y_train), n_folds=10, shuffle=True, random_state=21)
    cv_train.append(np.mean(cross_val_score(rfc, X_train, y_train, cv=k_fold)))
    print("in-sample accuracy: {}".format(np.mean(cross_val_score(rfc, 
          X_train, y_train, cv=k_fold))))
    print("out-of-sample accuracy: {}".format(accuracy_score(y_test,y_pred)))
    accuracy_test.append(accuracy_score(y_test,y_pred))
    print('\n')
    
#show in plot to observe more clearly
x=range(1,31)
plt.plot(x,cv_train,'bo')
plt.xlabel('n_estimators') 
plt.ylabel('in-sample accuracy') 
plt.show()
plt.plot(x,accuracy_test,'bo')
plt.xlabel('n_estimators') 
plt.ylabel('out-of-sample accuracy') 
plt.show()

#According to Part 1, I selected n_estimators as 8
#Because the accuracy is up to 1 at number 8&10
rfr = RandomForestClassifier(n_estimators=10,random_state=21)
rfr.fit(X,y) 
feat_labels = df_wine.columns[1:]
importances = rfr.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
        print("%2d: %-*s %f" % (f + 1, 30,feat_labels[f],importances[indices[f]]))
    

