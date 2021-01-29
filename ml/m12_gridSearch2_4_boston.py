# 모델 : RandomForestClassifier

parameters = [
    {'n_estimators':[100,200],'max_depth':[6,8,10,12],'min_samples_leaf':[3,5,7,10],
    'min_samples_split':[2,3,5,10],'n_jobs':[-1,2,4]},
    {'n_estimators':[300,400],'max_depth':[6,8,10,12]},
    {'n_estimators':[300,400],'min_samples_leaf':[3,5,7,10]},
    {'n_estimators':[100,200],'min_samples_split':[2,3,5,10]},
    {'max_depth':[2,3,5],'min_samples_leaf':[3,5,7,10],'n_jobs':[-1,2,4]}
]

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=32)

kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델 구성

model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold)
model.fit(x_train, y_train)

print('최적의 매개변수 :', model.best_estimator_)

y_pred = model.predict(x_test)
print('최종정답률 :',r2_score(y_test, y_pred))

# 최적의 매개변수 : RandomForestRegressor(max_depth=10, n_estimators=300)
# 최종정답률 : 0.8631149450042049

# 최적의 매개변수 : RandomForestRegressor(max_depth=12, n_estimators=400)
# 최종정답률 : 0.8620400325146376