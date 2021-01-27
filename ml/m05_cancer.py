import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 1. 데이터
dataset = load_breast_cancer()
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
# Linear SVC 0.9649122807017544
# SVC 0.9824561403508771
# KNeighborsClassifier 0.9736842105263158
# RandomForestClassifier 0.9649122807017544
# DecisionTreeClassifier 0.956140350877193 
# LogisticRegression 0.9736842105263158

# Standard
# Linear SVC 0.9736842105263158
# SVC 0.9736842105263158
# KNeighborsClassifier 0.9824561403508771
# RandomForestClassifier 0.956140350877193
# DecisionTreeClassifier 0.956140350877193
# LogisticRegression 0.9824561403508771

# Tensorflow
# acc :  0.9912280440330505