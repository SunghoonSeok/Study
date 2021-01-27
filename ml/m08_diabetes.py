import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성

model = DecisionTreeRegressor()

# 3. 컴파일, 훈련
model.fit(x_train, y_train)
# 4. 평가, 예측
y_pred = model.predict(x_test)
result = model.score(x_test, y_test)
print(result)
acc = r2_score(y_test, y_pred)
print(acc)

# MinMax
# Linear SVR 0.3072748836901975  0.3072748836901975
# SVR 0.1401655821255935  0.1401655821255935
# KNeighborsRegressor  0.40380234136299853  0.40380234136299853
# RandomForestRegressor 0.44254750464367854  0.4425475046436785
# DecisionTreeRegressor -0.08404261371933663  -0.08404261371933663

# Standard
# Linear SVR 0.4517761589389889  0.4517761589389889
# SVR 0.16028890079722924  0.16028890079722924
# KNeighborsRegressor 0.3255734870306528  0.3255734870306528
# RandomForestRegressor 0.42487762961558095  0.42487762961558095
# DecisionTreeRegressor -0.35932846024789167  -0.35932846024789167