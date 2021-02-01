from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
import pandas as pd
import numpy as np
# 1. 데이터
dataset = load_boston()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=32
)
# for i in [-1, 8, 4, 1]:
#     start = time.time()
#     model = XGBRegressor(n_jobs=i)
#     model.fit(x_train,y_train)
#     acc = model.score(x_test,y_test)
#     print('n_jobs가',f'{i}','일때')
#     print(time.time()-start,'초')


model = XGBRegressor(n_jobs=-1)

model.fit(x_train,y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc :", acc)

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
new_data=[]
feature=[]
a = np.percentile(model.feature_importances_, q=25)

for i in range(len(dataset.data[0])):
    if model.feature_importances_[i] > a:
       new_data.append(df.iloc[:,i])
       feature.append(dataset.feature_names[i])

new_data = pd.concat(new_data, axis=1)

        
x2_train, x2_test, y2_train, y2_test = train_test_split(new_data, dataset.target, train_size=0.8, random_state=32)
model2 = XGBRegressor(n_jobs=-1)   
model2.fit(x2_train,y2_train)
acc2 = model2.score(x2_test, y2_test)
print("acc2 :", acc2)
print(new_data.shape)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model, feature_name, data):
    n_features = data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_name)
    plt.xlabel("Feature Importances")
    plt.ylabel("features")
    plt.ylim(-1, n_features)
plot_feature_importances_dataset(model, dataset.feature_names, dataset.data)
plot_feature_importances_dataset(model2, feature, new_data)
plt.show()

# [0.01190844 0.00271136 0.01734594 0.00223939 0.04735148 0.16609913
#  0.0079637  0.0523662  0.02352589 0.0308157  0.10907637 0.00717398
#  0.5214224 ]
# acc : 0.8658592346452706
# acc2 : 0.8687695270691074
# (506, 9)


# n_jobs가 -1 일때
# 0.08475089073181152 초
# n_jobs가 8 일때
# 0.05684781074523926 초
# n_jobs가 4 일때
# 0.05884242057800293 초
# n_jobs가 1 일때
# 0.09377741813659668 초