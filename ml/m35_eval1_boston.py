# eval set

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

model = XGBRegressor(n_estimators=500, learning_rate=0.01, n_jobs=8)

model.fit(x_train, y_train, verbose=1, eval_metric='rmse', eval_set=[(x_train, y_train),(x_test,y_test)])

aaa = model.score(x_test, y_test)
print(aaa)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print("r2 :",r2)
print("==============================")

results = model.evals_result()
print(results)

# random_state=32
# r2 : 0.8720251352728053
# random_State=66
# r2 : 0.9329663244922279