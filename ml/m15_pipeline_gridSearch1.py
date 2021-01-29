import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=32)
parameters = [
    {"mal__C":[1, 10, 100, 1000], "mal__kernel":["linear"]},
    {"mal__C":[1, 10, 100], "mal__kernel":["rbf"],"mal__gamma":[0.001, 0.0001]},
    {"mal__C":[1, 10, 100, 1000], "mal__kernel":["sigmoid"],"mal__gamma":[0.001, 0.0001]}
]

# 2. 모델
pipe = Pipeline([("scaler", MinMaxScaler()),('mal', SVC())]) # 모델과 전처리를 한번에
# pipe = make_pipeline(StandardScaler(), SVC())
model = GridSearchCV(pipe, parameters, cv=5)



model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)
# 1.0
# 1.0