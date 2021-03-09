
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score


import pandas as pd

wine = pd.read_csv('c:/data/csv/winequality-white.csv', delimiter=';')

y= wine['quality']
x = wine.drop('quality', axis=1)

newlist=[]
for i in list(y):
    if i <=4:
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else:
        newlist +=[2]
y = newlist
# print(y)


parameters = [
    {'n_estimators':[100,200,300,400],'max_depth':[6,8,10,12,15],'min_samples_leaf':[3,5,7,10,12],
    'min_samples_split':[2,3,5,10,12]},
    {}
]


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델 구성

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold)
model.fit(x_train, y_train)

print('최적의 매개변수 :', model.best_estimator_)

y_pred = model.predict(x_test)
print('최종정답률 :',accuracy_score(y_test, y_pred))
