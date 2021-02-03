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
# train_data = train.copy()
# train_data = train_data.values
# x = train_data[:,3:]
# y1 = train_data[:,1]
# y2 = train_data[:,2]
# print(x.shape, y1.shape, y2.shape) 

temp = pd.DataFrame(train)
test_df = pd.DataFrame(test)
x = temp.iloc[:,3:]/255
y = temp.iloc[:,1]
x_test = test_df.iloc[:,2:]/255

x = x.to_numpy()
y1 = y.to_numpy()
x_pred = x_test.to_numpy()


x_train, x_test, y_train, y_test = train_test_split(x,y1, test_size=0.2, shuffle=True)

model = XGBClassifier(n_jobs=8, eval_metric='mlogloss', object='multi:softmax')

model.fit(x_train,y_train,verbose=True, eval_set=[(x_train,y_train),(x_test,y_test)])

#4. 평가, 예측

acc = model.score(x_test, y_test)
y_pred = model.predict(x_pred)




# model.save_model('../data/xgb_save/test5_mnist1.xgb.model')

# model2 = XGBRegressor()
# model2.load_model('../data/xgb_save/test5_mnist1.xgb.model')
# print("=============== xgb model 불러오기 ===============")
# print('불러왔다!')
# r22 = model2.score(x_test, y_test)
# print('r22 :',r22)
submission.iloc[:,1] = y_pred.reshape(-1,1)
submission.to_csv('c:/data/test/mnist/submission_mnist1.csv', index=False)  