# diabetes, RandomForest
# 파이프라인 엮어서 25번 돌리기


import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

import warnings
import time
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target


kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {'mal__n_estimators':[100,200],'mal__min_samples_leaf':[3,5],
    'mal__min_samples_split':[2,5],'mal__n_jobs':[-1]},
    {'mal__n_estimators':[300,400],'mal__max_depth':[6,8]}
    # {'mal__max_depth':[2,3,5],'mal__min_samples_leaf':[3,5,7,10]},
    # {'mal__n_estimators':[100,200],'mal__min_samples_split':[2,3,5,10]},
    # {'mal__n_estimators':[300,400],'mal__max_depth':[2,3,5],'mal__n_jobs':[-1]},
    # {}
]

# 2. 모델 구성

i = 1
scalers = [MinMaxScaler(), StandardScaler()]
search = [GridSearchCV, RandomizedSearchCV]


kfold = KFold(n_splits=5, shuffle=True)

a=[]

i = 1
for train_index, test_index in kfold.split(x):
    print(str(i)+'번째 kfold split')
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    pipe = Pipeline([("scaler", MinMaxScaler()),('mal', RandomForestRegressor())])
    model = RandomizedSearchCV(pipe, parameters)
    score = cross_val_score(model, x_train, y_train, cv=kfold)
    print("scores :",score)
    a.append(score)
    i += 1

# 1번째 kfold split
# scores : [0.4502274  0.31406065 0.33323133 0.54070153 0.44111037]
# 2번째 kfold split
# scores : [0.54278944 0.25124213 0.37718554 0.47460629 0.45400059]
# 3번째 kfold split
# scores : [0.54559054 0.5922048  0.46075512 0.31789233 0.41273222]
# 4번째 kfold split
# scores : [0.42247993 0.39015017 0.50966969 0.55285347 0.44333145]
# 5번째 kfold split
# scores : [0.44477028 0.50576677 0.42131651 0.42976234 0.39110624]