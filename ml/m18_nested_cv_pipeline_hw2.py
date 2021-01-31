# cancer, RandomForest
# 파이프라인 엮어서 25번 돌리기


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

import warnings
import time
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_breast_cancer()
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
    pipe = Pipeline([("scaler", MinMaxScaler()),('mal', RandomForestClassifier())])
    model = RandomizedSearchCV(pipe, parameters)
    score = cross_val_score(model, x_train, y_train, cv=kfold)
    print("scores :",score)
    a.append(score)
    i += 1

# 1번째 kfold split
# scores : [0.97802198 0.9010989  0.95604396 0.96703297 0.95604396]
# 2번째 kfold split
# scores : [0.92307692 0.95604396 0.98901099 0.96703297 0.95604396]
# 3번째 kfold split
# scores : [0.98901099 0.96703297 0.94505495 0.96703297 0.94505495]
# 4번째 kfold split
# scores : [0.97802198 0.97802198 0.95604396 0.95604396 0.94505495]
# 5번째 kfold split
# scores : [0.95652174 0.93406593 0.97802198 0.96703297 0.95604396]