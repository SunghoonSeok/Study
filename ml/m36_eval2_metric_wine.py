# eval set

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

x, y = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

model = XGBClassifier(n_estimators=500, learning_rate=0.01, n_jobs=8, eval_metric=['mlogloss','merror'])

model.fit(x_train, y_train, verbose=1, eval_set=[(x_train, y_train),(x_test,y_test)])

aaa = model.score(x_test, y_test)
print(aaa)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("acc :",acc)
print("==============================")

results = model.evals_result()
print(results)

# 1.0
# acc : 1.0