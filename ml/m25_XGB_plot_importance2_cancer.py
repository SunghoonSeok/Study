from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import pandas as pd
import numpy as np
import time
# 1. 데이터
dataset = load_breast_cancer()
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
model2 = XGBClassifier(n_jobs=-1)   
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

# [0.         0.01134981 0.00311594 0.01064076 0.00151385 0.00158364
#  0.00937885 0.02125592 0.00126252 0.00533811 0.01396505 0.01638373
#  0.00610274 0.0060422  0.01029904 0.00240293 0.00201137 0.00537911
#  0.00552537 0.00140438 0.5223647  0.01457605 0.22440161 0.01400607
#  0.00642338 0.00601382 0.01596073 0.0577923  0.00350612 0.        ]
# acc : 0.956140350877193
# acc2 : 0.956140350877193
# (569, 22)



# n_jobs가 -1 일때
# 0.06080818176269531 초
# n_jobs가 8 일때
# 0.03291177749633789 초
# n_jobs가 4 일때
# 0.03194379806518555 초
# n_jobs가 1 일때
# 0.06479263305664062 초