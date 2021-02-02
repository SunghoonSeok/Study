
parameters = [
    {"mal__n_estimators":[100,200,300],"mal__learning_rate":[0.1,0.3,0.5], "mal__max_depth":[4,5,6]},
    {"mal__n_estimators":[90,100,110],"mal__learning_rate":[0.1,0.01,0.001],"mal__max_depth":[4,5,6],
    "mal__colsample_bytree":[0.6,0.9,1]},
    {"mal__n_estimators":[90,110],"mal__learning_rate":[0.1,0.001,0.5],"mal__max_depth":[4,5,6],"mal__colsample_bytree":[0.6,0.9,1],
    "mal__colsample_bylevel":[0.6,0.7,0.9]}
]
import warnings
import pandas as pd
import time
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
warnings.filterwarnings('ignore')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)
x = x.reshape(70000, 28*28)

# pca = PCA()
# x = pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum :",cumsum)

# d = np.argmax(cumsum >=1)+1
# print("cumsum >=0.95 :", cumsum>=0.95)
# print("d :", d) # d : 674 # 714

pca = PCA(n_components=154)
x2 = pca.fit_transform(x)

x2_train = x2[:60000,:]
x2_test = x2[60000:,:]

kfold = KFold(n_splits=5, shuffle=True)

start = time.time()
scalers = [MinMaxScaler(), StandardScaler()]
models = [XGBClassifier(eval_metric='mlogloss', n_jobs=8)]
search = [GridSearchCV, RandomizedSearchCV]
for i in scalers:
    print(f'{i}')
    for j in models:
        pipe = Pipeline([("scaler", i),('mal', j)])
        for k in search:
            model = k(pipe, parameters, cv=kfold)
            model.fit(x2_train, y_train)
            result = model.score(x2_test, y_test)
            y_pred = model.predict(x2_test)
            print(f'{k}')
            print(result)
            print('최적의 매개변수 :', model.best_estimator_)
            print('걸린 시간 :', time.time()-start, '초')
            print('최종정답률 :',accuracy_score(y_test, y_pred))
