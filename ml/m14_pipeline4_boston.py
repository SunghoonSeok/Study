import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline

# 1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=32)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델
# model = Pipeline([("scaler", MinMaxScaler()),('malddong', SVC())]) # 모델과 전처리를 한번에
# model = make_pipeline(StandardScaler(), RandomForestRegressor())
# model.fit(x_train, y_train)

# result = model.score(x_test, y_test)
# print(result)

scalers = [MinMaxScaler(), StandardScaler()]
models = [RandomForestRegressor(),DecisionTreeRegressor(),KNeighborsRegressor()]
for i in scalers:
    print(f'{i}')
    for j in models:
        model = make_pipeline(i, j)
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        print(result,'-' ,f'{j}')


# 0.906832
# 0.92374

# MinMaxScaler()
# 0.90064 - RandomForestRegressor()
# 0.72 - DecisionTreeRegressor()
# 0.9392 - KNeighborsRegressor()
# StandardScaler()
# 0.902828 - RandomForestRegressor()
# 0.6799999999999999 - DecisionTreeRegressor()
# 0.9536 - KNeighborsRegressor()