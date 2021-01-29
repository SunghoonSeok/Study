import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=32)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"]},
    {"C":[1, 10, 100], "kernel":["rbf"],"gamma":[0.001, 0.0001]},
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],"gamma":[0.001, 0.0001]}
]

# 2. 모델 구성

model = GridSearchCV(SVC(), parameters, cv=kfold)
score =cross_val_score(model, x_train, y_train, cv=kfold)

print('교차검증점수 :',score)

# model.fit(x_train, y_train)

# print('최적의 매개변수 :', model.best_estimator_)

# y_pred = model.predict(x_test)
# print('최종정답률 :',accuracy_score(y_test, y_pred))

# # 최적의 매개변수 : SVC(C=1, kernel='linear')
# # 최종정답률 : 1.0