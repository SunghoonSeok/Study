import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=32, shuffle=True, train_size=0.8)

kfold = KFold(n_splits=5, shuffle=True)


# 2. 모델 구성

model = [LinearSVR(),SVR(),KNeighborsRegressor(),RandomForestRegressor(),DecisionTreeRegressor(), LinearRegression()]
for i in model:
    score = cross_val_score(i, x_train, y_train, cv=kfold)
    print(f'\n{i}')
    print('scores :', score)

# LinearSVR()
# scores : [ 0.50177111  0.5399789   0.38071459  0.38615419 -1.07525117]

# SVR()
# scores : [0.3317449  0.27143328 0.09960354 0.28384842 0.1759996 ]

# KNeighborsRegressor()
# scores : [0.49184374 0.33949719 0.60401271 0.62879381 0.37401817]

# RandomForestRegressor()
# scores : [0.89186644 0.86682337 0.85872988 0.82610344 0.92595461]

# DecisionTreeRegressor()
# scores : [0.82796758 0.55530162 0.63714395 0.82927361 0.72302139]

# LinearRegression()
# scores : [0.78321407 0.73891065 0.65567153 0.79622583 0.63766697]