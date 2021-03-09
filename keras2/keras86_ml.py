from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

import pandas as pd

df = pd.read_csv('c:/data/csv/winequality-white.csv', delimiter=';')
print(df.head())
print(df.index)

print(df.shape) # 4898,12

x = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(x.shape, y.shape)

print(y.unique()) # 7

x = x.values
y = y.values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=66)

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
scaler = QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

model = XGBRegressor(n_jobs=8)
model.fit(x_train, y_train)
score = model.score(x_test,y_test)
print("R2 :", score)

thresholds = np.sort(model.feature_importances_)
print(thresholds)


for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) # prefit에 대해 알아볼것 과제!
    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)
    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%"%(thresh,select_x_train.shape[1],score*100))
