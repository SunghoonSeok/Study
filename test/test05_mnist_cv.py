import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline, make_pipeline
import warnings

warnings.filterwarnings('ignore')
train = read_csv('../data/test/mnist/train.csv', index_col=None, header=0)
test = read_csv('../data/test/mnist/test.csv', index_col=None, header=0)
submission = read_csv('../data/test/mnist/submission.csv', index_col=None, header=0)


temp = pd.DataFrame(train)
test_df = pd.DataFrame(test)
x = temp.iloc[:,3:]/255
y = temp.iloc[:,1]
y2 = temp.iloc[:,2]
x_test = test_df.iloc[:,2:]/255

x = x.to_numpy()
y = y.to_numpy()
x_pred = x_test.to_numpy()

pca = PCA(n_components=147)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y1, test_size=0.2, shuffle=True)
kfold = KFold(n_splits=5, shuffle=True)


start = time.time()
model = XGBClassifier(n_jobs=8, eval_metric='mlogloss', n_estimators=1000, early_stopping_rounds=10,
                     ) # use_label_encoder=False)

model = k(models, parameters, cv=kfold)
model.fit(x_train, y_train, verbose=True, eval_set=[(x_train,y_train),(x_test,y_test)])
result = model.score(x_test, y_test)
y_pred = model.predict(x_test)
print(f'{k}')
print(result)
print('최적의 매개변수 :', model.best_estimator_)
print('걸린 시간 :', time.time()-start, '초')
print('최종정답률 :',accuracy_score(y_test, y_pred))