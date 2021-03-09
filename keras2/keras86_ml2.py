from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import xgboost
from xgboost import XGBClassifier, XGBRFClassifier

warnings.filterwarnings('ignore')

import pandas as pd

df = pd.read_csv('c:/data/csv/winequality-white.csv', delimiter=';')
print(df.corr)
print(df.head())
print(df.index)

print(df.shape) # 4898,12

x = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(x.shape, y.shape)

print(y.unique()) # 7

x = x.values
y = y.values

from sklearn.preprocessing import OneHotEncoder

# y = y.reshape(-1,1)

# ohencoder = OneHotEncoder()
# ohencoder.fit(y)
# y = ohencoder.transform(y).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
print(score)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print(score)