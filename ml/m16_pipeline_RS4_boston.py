# 실습
# RandomSearch, GS와 pipeline을 엮어라
# 모델은 RandomForest


import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
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
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=32)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {'mal__n_estimators':[100,200],'mal__max_depth':[6,8,10,12],'mal__min_samples_leaf':[3,5,7,10],
    'mal__min_samples_split':[2,3,5,10],'mal__n_jobs':[-1]},
    {'mal__n_estimators':[300,400],'mal__max_depth':[6,8,10,12]},
    {'mal__n_estimators':[300,400],'mal__max_depth':[2,3,5],'mal__min_samples_leaf':[3,5,7,10]},
    {'mal__n_estimators':[100,200],'mal__min_samples_split':[2,3,5,10]},
    {'mal__n_estimators':[300,400],'mal__max_depth':[2,3,5],'mal__min_samples_leaf':[3,5,7,10],'mal__n_jobs':[-1]}
]


# 2. 모델 구성
start = time.time()
scalers = [MinMaxScaler(), StandardScaler()]
models = [RandomForestRegressor()]
search = [GridSearchCV, RandomizedSearchCV]
for i in scalers:
    print(f'{i}')
    for j in models:
        pipe = Pipeline([("scaler", i),('mal', j)])
        for k in search:
            model = k(pipe, parameters, cv=kfold)
            model.fit(x_train, y_train)
            result = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            print(f'{k.__name__}')
            print(result,'-' ,f'{j}')
            print('최적의 매개변수 :', model.best_estimator_)
            print('걸린 시간 :', time.time()-start, '초')
            print('최종정답률 :',r2_score(y_test, y_pred))
            print('\n')


# MinMaxScaler()
# GridSearchCV
# 0.8681958464487642 - RandomForestRegressor()
# 최적의 매개변수 : Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('mal', RandomForestRegressor(max_depth=12, n_estimators=400))])
# 걸린 시간 : 163.0705053806305 초
# 최종정답률 : 0.8681958464487642
# RandomizedSearchCV
# 0.8569571002947596 - RandomForestRegressor()
# 최적의 매개변수 : Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('mal', RandomForestRegressor(min_samples_split=10))])
# 걸린 시간 : 172.60608553886414 초
# 최종정답률 : 0.8569571002947596
# StandardScaler()
# GridSearchCV
# 0.8569240282793105 - RandomForestRegressor()
# 최적의 매개변수 : Pipeline(steps=[('scaler', StandardScaler()),
#                 ('mal', RandomForestRegressor(max_depth=12, n_estimators=400))])
# 걸린 시간 : 334.9474537372589 초
# 최종정답률 : 0.8569240282793105
# RandomizedSearchCV
# 0.8560618632130265 - RandomForestRegressor()
# 최적의 매개변수 : Pipeline(steps=[('scaler', StandardScaler()),
#                 ('mal',
#                  RandomForestRegressor(min_samples_split=3, n_estimators=200))])
# 걸린 시간 : 343.5400884151459 초
# 최종정답률 : 0.8560618632130265





