import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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


x_train, x_Test, y_train, y_test = train_test_split(x, y, random_state=32, shuffle=True, train_size=0.8)

kfold = KFold(n_splits=5, shuffle=True)


# 2. 모델 구성

model = LogisticRegression()
score = cross_val_score(model, x_train, y_train, cv=kfold)
print('scores :', score)

'''
# 3. 컴파일, 훈련
model.fit(x_train, y_train)
# 4. 평가, 예측
y_pred = model.predict(x_test)
result = model.score(x_test, y_test)
print(result)
acc = accuracy_score(y_test, y_pred)
print(acc)

'''
# scores : [0.91666667 0.95833333 0.95833333 0.95833333 1.        ]