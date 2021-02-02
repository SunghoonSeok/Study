# eval set

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

model = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=8)

model.fit(x_train, y_train, verbose=1, eval_metric=['rmse','logloss','mae'],
          early_stopping_rounds=10, eval_set=[(x_train, y_train),(x_test,y_test)])

aaa = model.score(x_test, y_test)
print("model.score :",aaa)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 :",r2)
print("==============================")

results = model.evals_result()
# print(results)

import pickle
# pickle.dump(model, open('../data/xgb_save/m39.pickle.dat', 'wb'))
# print('저장완료')
import joblib
# joblib.dump(model, '../data/xgb_save/m39.joblib.dat')
print("=============== joblib 불러오기 ===============")
model2 = joblib.load('../data/xgb_save/m39.joblib.dat')
print('불러왔다!')
r22 = model2.score(x_test, y_test)
print('r22 :', r22)

# r2 : 0.93302293398985
# ==============================
# =============== joblib 불러오기 ===============
# 불러왔다!
# r22 : 0.93302293398985
'''
print("=============== pickle 불러오기 ===============")

# 불러오기
model2 = pickle.load(open('../data/xgb_save/m39.pickle.dat', 'rb'))
print('불러왔다!')
r22 = model2.score(x_test, y_test)
print('r22 :', r22)

# model.score : 0.93302293398985
# r2 : 0.93302293398985
# ==============================
# =============== pickle 불러오기 ===============
# 불러왔다!
# r22 : 0.93302293398985
'''