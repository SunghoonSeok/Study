import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import warnings
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
warnings.filterwarnings('ignore')

from pandas import read_csv
train = read_csv('../data/test/mnist/train.csv', index_col=None, header=0)
test = read_csv('../data/test/mnist/test.csv', index_col=None, header=0)
submission = read_csv('../data/test/mnist/submission.csv', index_col=None, header=0)

print(train.shape) # (2048, 787)
print(test.shape) # (20480, 786)
print(submission.shape) # (20480, 2)
train_data = train.copy()
train_data = train_data.values
x = train_data[:,3:]
y1 = train_data[:,1]
y2 = train_data[:,2]
print(x.shape, y1.shape, y2.shape) 

# pca = PCA()
# x = pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum :",cumsum)

# d = np.argmax(cumsum >=0.97)+1
# print("cumsum >=0.95 :", cumsum>=0.97)
# print("d :", d) # 147

pca = PCA(n_components=147)
x = pca.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y1, test_size=0.2, shuffle=True,random_state=66)

model = XGBClassifier(n_jobs=8, eval_metric='mlogloss')

model.fit(x_train,y_train,verbose=True, eval_set=[(x_train,y_train),(x_test,y_test)])

print(x_test.shape, y_test.shape)

#4. 평가, 예측
accuracy = model.score(x_test, y_test)

print("acc :", accuracy)
