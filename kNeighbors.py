import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    # used to split the dataset
from sklearn.datasets import load_iris    # loading the iris dataset
from sklearn.neighbors import KNeighborsClassifier    # to check the nearby neighbors features and similarities
from sklearn.metrics import confusion_matrix   # used to get the accuracy and the prediction of the model
datasets = load_iris()   # assigning the datasets
x = datasets.data
y = datasets.target
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state = 0)
kn = KNeighborsClassifier(n_neighbors = 3)  # model under test
kn.fit(train_x, train_y)
y_predict = kn.predict(test_x)
print(y_predict)
print(test_y)
cm = confusion_matrix(test_y, y_predict)
print(cm)
