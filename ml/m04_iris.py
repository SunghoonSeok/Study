import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) #(150, 4),  (150, )
print(x[:5])
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성

model = LogisticRegression()


# 3. 컴파일, 훈련
model.fit(x_train, y_train)
# 4. 평가, 예측
y_pred = model.predict(x_test)
result = model.score(x_test, y_test)
print(result)
acc = accuracy_score(y_test, y_pred)
print(acc)

# MinMax
# Linear SVC 0.9
# SVC 0.9333333333333333
# KNeighborsClassifier 0.9333333333333333
# RandomForestClassifier 0.9666666666666667
# DecisionTreeClassifier 0.9666666666666667 
# LogisticRegression 0.9666666666666667

# Standard
# Linear SVC 0.9333333333333333  ~ 0.9666666666666667
# SVC 0.9333333333333333 ~ 1.0
# KNeighborsClassifier 0.9333333333333333 ~ 1.0
# RandomForestClassifier 0.9666666666666667 ~ 1.0
# DecisionTreeClassifier 0.9666666666666667 ~ 1.0
# LogisticRegression 0.9666666666666667

# Tensorflow
# 1.0