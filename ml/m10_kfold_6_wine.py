import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=32, shuffle=True, train_size=0.8)

kfold = KFold(n_splits=5, shuffle=True)


# 2. 모델 구성

model = [LinearSVC(),SVC(),KNeighborsClassifier(),RandomForestClassifier(),DecisionTreeClassifier(), LogisticRegression()]
for i in model:
    score = cross_val_score(i, x_train, y_train, cv=kfold)
    print(f'\n{i}')
    print('scores :', score)

# LinearSVC()
# scores : [0.94505495 0.94505495 0.73626374 0.92307692 0.93406593]

# SVC()
# scores : [0.94505495 0.93406593 0.94505495 0.89010989 0.92307692]

# KNeighborsClassifier()
# scores : [0.96703297 0.94505495 0.86813187 0.97802198 0.92307692]

# RandomForestClassifier()
# scores : [0.98901099 0.98901099 0.96703297 0.95604396 0.91208791]

# DecisionTreeClassifier()
# scores : [0.92307692 0.96703297 0.93406593 0.98901099 0.94505495]

# LogisticRegression()
# scores : [0.93103448 1.         0.92857143 0.96428571 0.92857143]