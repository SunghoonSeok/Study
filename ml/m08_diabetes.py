import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# 1. 데이터
dataset = load_diabetes()
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
# model.score : 0.40729129662251917
# r2_score : 0.40729129662251917

# RandomForestRegressor()
# model.score : 0.42481851199779685
# r2_score : 0.42481851199779685

# DecisionTreeRegressor()
# model.score : -0.11309841650939156
# r2_score : -0.11309841650939156

# KNeighborsRegressor()
# model.score : 0.33049065564097546
# r2_score : 0.33049065564097546

# Standard
# LinearRegression()
# model.score : 0.5697064035593562
# r2_score : 0.5697064035593562

# RandomForestRegressor()
# model.score : 0.5310880675967731
# r2_score : 0.5310880675967731

# DecisionTreeRegressor()
# model.score : -0.05170994009999097
# r2_score : -0.05170994009999097

# KNeighborsRegressor()
# model.score : 0.48762179487052615
# r2_score : 0.48762179487052615

# Tensorflow
# R2 :  0.5208084824108226