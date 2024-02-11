import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

iris = sns.load_dataset('iris')

print(
        iris.info()
        )

# Train Test Split

X = iris.drop('species', axis = 1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

model = SVC()

# fit the model
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

'''
Confusion_Matrix
[[13  0  0]
 [ 0 19  1]
 [ 0  0 12]]


classification_report
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        13
  versicolor       1.00      0.95      0.97        20
   virginica       0.92      1.00      0.96        12

    accuracy                           0.98        45
   macro avg       0.97      0.98      0.98        45
weighted avg       0.98      0.98      0.98        45
'''

param_grid = {'C': [0.1, 1 , 10 ,100, 1000], 'gamma': [1,0.1, 0.01, 0.001, 0.0001]}

grid = GridSearchCV(model, param_grid, verbose = 3)

grid.fit(X_train, y_train)

grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test, grid_predictions))
print('\n')
print(classification_report(y_test, grid_predictions))

'''
Confusion_Matrix
[[13  0  0]
 [ 0 19  1]
 [ 0  0 12]]

classification_report

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        13
  versicolor       1.00      0.95      0.97        20
   virginica       0.92      1.00      0.96        12

    accuracy                           0.98        45
   macro avg       0.97      0.98      0.98        45
weighted avg       0.98      0.98      0.98        45
'''
