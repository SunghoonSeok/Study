import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=32, shuffle=True, train_size=0.8)

kfold = KFold(n_splits=5, shuffle=True)


# 2. 모델 구성

model = [LinearSVC(),SVC(),KNeighborsClassifier(),RandomForestClassifier(),DecisionTreeClassifier()]
for i in model:
    score = cross_val_score(i, x_train, y_train, cv=kfold)
    print(f'\n{i}')
    print('scores :', score)

# LinearSVC()
# scores : [0.95833333 1.         1.         0.91666667 0.91666667]

# SVC()
# scores : [1.         0.91666667 1.         0.91666667 0.91666667]

# KNeighborsClassifier()
# scores : [0.95833333 1.         0.95833333 0.875      0.95833333]

# RandomForestClassifier()
# scores : [0.95833333 0.875      0.95833333 0.95833333 0.91666667]

# DecisionTreeClassifier()
# scores : [0.875      0.95833333 0.95833333 0.95833333 0.95833333] 

# LogisticRegression()
# scores : [0.91666667 0.95833333 0.875      1.         0.95833333]