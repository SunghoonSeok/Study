
parameters = [
    {"mal__n_estimators":[100,200,300],"mal__learning_rate":[0.1,0.3,0.5], "mal__max_depth":[4,5,6]},
    {"mal__n_estimators":[90,100,110],"mal__learning_rate":[0.1,0.01,0.001],"mal__max_depth":[4,5,6],
    "mal__colsample_bytree":[0.6,0.9,1]},
    {"mal__n_estimators":[90,110],"mal__learning_rate":[0.1,0.001,0.5],"mal__max_depth":[4,5,6],"mal__colsample_bytree":[0.6,0.9,1],
    "mal__colsample_bylevel":[0.6,0.7,0.9]}
]
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
import warnings
import pandas as pd
import numpy as np
import time
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=32
)
kfold = KFold(n_splits=5, shuffle=True)

start = time.time()
scalers = [MinMaxScaler(), StandardScaler()]
models = [XGBClassifier(eval_metric='mlogloss')]
search = [GridSearchCV]
for i in scalers:
    print(f'{i}')
    for j in models:
        pipe = Pipeline([("scaler", i),('mal', j)])
        for k in search:
            model = k(pipe, parameters, cv=kfold)
            model.fit(x_train, y_train)
            result = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            print(f'{k}')
            print(result,'-' ,f'{j}')
            print('최적의 매개변수 :', model.best_estimator_)
            print('걸린 시간 :', time.time()-start, '초')
            print('최종정답률 :',accuracy_score(y_test, y_pred))

# MinMaxScaler()
# <class 'sklearn.model_selection._search.GridSearchCV'>
# 0.9666666666666667 
# 최적의 매개변수 : Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('mal',
#                  XGBClassifier(base_score=0.5, booster='gbtree',
#                                colsample_bylevel=1, colsample_bynode=1,
#                                colsample_bytree=0.6, eval_metric='mlogloss',
#                                gamma=0, gpu_id=-1, importance_type='gain',
#                                interaction_constraints='', learning_rate=0.01,
#                                max_delta_step=0, max_depth=4,
#                                min_child_weight=1, missing=nan,
#                                monotone_constraints='()', n_estimators=90,
#                                n_jobs=8, num_parallel_tree=1,
#                                objective='multi:softprob', random_state=0,
#                                reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
#                                subsample=1, tree_method='exact',
#                                validate_parameters=1, verbosity=None))])
# 걸린 시간 : 56.98653554916382 초
# 최종정답률 : 0.9666666666666667
# StandardScaler()
# <class 'sklearn.model_selection._search.GridSearchCV'>
# 0.9333333333333333 
# 최적의 매개변수 : Pipeline(steps=[('scaler', StandardScaler()),
#                 ('mal',
#                  XGBClassifier(base_score=0.5, booster='gbtree',
#                                colsample_bylevel=0.6, colsample_bynode=1,
#                                colsample_bytree=0.6, eval_metric='mlogloss',
#                                gamma=0, gpu_id=-1, importance_type='gain',
#                                interaction_constraints='', learning_rate=0.1,
#                                max_delta_step=0, max_depth=4,
#                                min_child_weight=1, missing=nan,
#                                monotone_constraints='()', n_estimators=110,
#                                n_jobs=8, num_parallel_tree=1,
#                                objective='multi:softprob', random_state=0,
#                                reg_alpha=0, reg_lambda=1, scale_pos_weight=None,
#                                subsample=1, tree_method='exact',
#                                validate_parameters=1, verbosity=None))])
# 걸린 시간 : 117.52357268333435 초
# 최종정답률 : 0.9333333333333333