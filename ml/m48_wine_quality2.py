
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
count_data = wine.groupby('quality')['quality'].count()
print(count_data)
x = wine.iloc[:,:-1]
y = wine.iloc[:,-1]

x = x.values
y = y.values

parameters = [
    {'n_estimators':[100,200,300,400],'max_depth':[6,8,10,12,15],'min_samples_leaf':[3,5,7,10,12],
    'min_samples_split':[2,3,5,10,12]}
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
model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print('최종정답률 :',accuracy_score(y_test, y_pred))
thresholds = np.sort(model.feature_importances_)
print(thresholds)
# 최종정답률 : 0.7030612244897959
# [0.0742786  0.08188823 0.08194727 0.0857064  0.08651896 0.08751794
#  0.08976668 0.09270981 0.09814083 0.10380411 0.11772117]

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, n_iter=20)
model.fit(x_train, y_train)

print('최적의 매개변수 :', model.best_estimator_)

from sklearn.feature_selection import SelectFromModel

for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True) # prefit에 대해 알아볼것 과제!
    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)
    selection_model = RandomForestClassifier(max_depth=15, min_samples_leaf=3, min_samples_split=10,n_estimators=200)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = accuracy_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%"%(thresh,select_x_train.shape[1],score*100))
