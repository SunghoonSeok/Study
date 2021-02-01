from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time
import pandas as pd
import numpy as np
# 1. 데이터
dataset = load_wine()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=32
)
# for i in [-1, 8, 4, 1]:
#     start = time.time()
#     model = XGBClassifier(n_jobs=i)
#     model.fit(x_train,y_train)
#     acc = model.score(x_test,y_test)
#     print('n_jobs가',f'{i}','일때')
#     print(time.time()-start,'초')




model = XGBClassifier(n_jobs=-1)

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
model2 =XGBClassifier(n_jobs=-1)   
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
# plot_feature_importances_dataset(model2, feature, new_data)
plt.show()

# [0.01793773 0.07281481 0.01100753 0.04728011 0.00852849 0.00979971
#  0.0569862  0.         0.03116941 0.13856803 0.01974412 0.43692654
#  0.14923736]
# acc : 0.9722222222222222
# acc2 : 0.9722222222222222
# (178, 9)


# n_jobs가 -1 일때
# 0.061806440353393555 초
# n_jobs가 8 일때       
# 0.03390908241271973 초
# n_jobs가 4 일때
# 0.03291201591491699 초
# n_jobs가 1 일때
# 0.04089069366455078 초