from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# 1. 데이터
dataset = load_wine()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=32
)

model = GradientBoostingClassifier(max_depth=4)

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
model2 = GradientBoostingClassifier(max_depth=4)   
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

plot_feature_importances_dataset(model2, feature, new_data)
plt.show()

# [8.97623933e-03 8.34847101e-02 9.69793317e-03 5.53122682e-03
#  5.98013775e-03 2.86370412e-03 5.24692578e-02 9.62849296e-08
#  5.70786024e-03 2.66222110e-01 1.82703746e-02 2.42900482e-01
#  2.97895867e-01]
# acc : 0.9444444444444444
# acc2 : 0.9444444444444444
# (178, 9)