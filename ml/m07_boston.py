import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
models =[LinearRegression(), RandomForestRegressor(), DecisionTreeRegressor(), KNeighborsRegressor()]
for i in models:
    model = i
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    result = model.score(x_test, y_test)
    print(f'\n{i}')
    print("model.score :",result)
    acc = r2_score(y_test, y_pred)
    print("r2_score :",acc)

# MinMax
# LinearRegression()
# model.score : 0.6648002998142213
# r2_score : 0.6648002998142213

# RandomForestRegressor()
# model.score : 0.8859948974700671
# r2_score : 0.8859948974700671

# DecisionTreeRegressor()
# model.score : 0.7834901613329981
# r2_score : 0.7834901613329981

# KNeighborsRegressor()
# model.score : 0.7326477193079846
# r2_score : 0.7326477193079846


# Standard
# LinearRegression()
# model.score : 0.7796055991479827
# r2_score : 0.7796055991479827

# RandomForestRegressor()
# model.score : 0.876522103679384
# r2_score : 0.876522103679384

# DecisionTreeRegressor()
# model.score : 0.6541205205234828
# r2_score : 0.6541205205234828

# KNeighborsRegressor()
# model.score : 0.8085705151414402
# r2_score : 0.8085705151414402

# Tensorflow
# R2 :  0.8907404253138521