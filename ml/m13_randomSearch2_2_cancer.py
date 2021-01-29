# 모델 : RandomForestClassifier

parameters = [
    {'n_estimators':[100,200],'max_depth':[6,8,10,12],'min_samples_leaf':[3,5,7,10],
    'min_samples_split':[2,3,5,10],'n_jobs':[-1]},
    {'n_estimators':[300,400],'max_depth':[6,8,10,12]},
    {'n_estimators':[300,400],'min_samples_leaf':[3,5,7,10]},
    {'n_estimators':[100,200],'min_samples_split':[2,3,5,10]},
    {'max_depth':[2,3,5],'min_samples_leaf':[3,5,7,10],'n_jobs':[-1]}
]

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=32)

kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델 구성

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold)
model.fit(x_train, y_train)

print('최적의 매개변수 :', model.best_estimator_)

y_pred = model.predict(x_test)
print('최종정답률 :',accuracy_score(y_test, y_pred))



# 최적의 매개변수 : RandomForestClassifier(min_samples_split=3, n_estimators=200)
# 최종정답률 : 0.9473684210526315

# 최적의 매개변수 : RandomForestClassifier(max_depth=8, min_samples_leaf=5, min_samples_split=10,
#                        n_jobs=-1)
# 최종정답률 : 0.9210526315789473