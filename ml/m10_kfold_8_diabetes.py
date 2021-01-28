import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# 1. 데이터
dataset = load_diabetes()
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
# scores : [-0.60759558 -0.50666705 -0.5357063  -0.59189867 -0.24751848]

# SVR()
# scores : [0.12462934 0.15172725 0.04347415 0.14119434 0.14284066]

# KNeighborsRegressor()
# scores : [0.45436719 0.39072902 0.37362411 0.30869334 0.42056613]

# RandomForestRegressor()
# scores : [0.4368787  0.40133147 0.44995187 0.37570559 0.39474203]

# DecisionTreeRegressor()
# scores : [-0.1133934  -0.17219447 -0.39983092 -0.06480348  0.11784658]

# LinearRegression()
# scores : [0.38977216 0.54322009 0.31239328 0.5885508  0.51938403]