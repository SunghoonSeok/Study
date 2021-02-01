from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
import warnings
import pandas as pd
import numpy as np
import time
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=32
)

# for i in [-1, 8, 4, 1]:
#     start = time.time()
#     model = XGBClassifier(n_jobs=i)
#     model.fit(x_train,y_train)
#     acc = model.score(x_test,y_test)
#     print("acc :", acc)
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
model2 = XGBClassifier(n_jobs=-1)   
model2.fit(x2_train,y2_train)
acc2 = model2.score(x2_test, y2_test)
print("acc2 :", acc2)
print(new_data.shape)

import matplotlib.pyplot as plt
import numpy as np
'''
def plot_feature_importances_dataset(model, feature_name, data):
    n_features = data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_name)
    plt.xlabel("Feature Importances")
    plt.ylabel("features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model, dataset.feature_names, dataset.data)
# plot_feature_importances_dataset(model2, feature, new_data)
'''
plot_importance(model)
plt.show()

# [0.01310219 0.01517014 0.5371727  0.43455493]
# acc : 0.9666666666666667
# acc2 : 0.9666666666666667
# (150, 3)


# n_jobs가 -1 일때
# 0.07177495956420898 초

# n_jobs가 8 일때
# 0.039893388748168945 초

# n_jobs가 4 일때
# 0.04288530349731445 초

# n_jobs가 1 일때
# 0.032911062240600586 초