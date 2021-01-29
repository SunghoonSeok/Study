import numpy as np
from sklearn.datasets import load_diabetes
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
dataset = load_diabetes()
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

# 0.45789968470035125
# 0.46550635970058674
scalers = [MinMaxScaler(), StandardScaler()]
models = [RandomForestRegressor(),DecisionTreeRegressor(),KNeighborsRegressor()]
for i in scalers:
    print(f'{i}')
    for j in models:
        model = make_pipeline(i, j)
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        print(result,'-' ,f'{j}')

# MinMaxScaler()
# 0.4476922412148985 - RandomForestRegressor()
# -0.17006119442750656 - DecisionTreeRegressor()
# 0.4435641769984564 - KNeighborsRegressor()
# StandardScaler()
# 0.4469073453009933 - RandomForestRegressor()
# -0.34224635743320775 - DecisionTreeRegressor()
# 0.4693915981946304 - KNeighborsRegressor()