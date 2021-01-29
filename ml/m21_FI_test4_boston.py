from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# 1. 데이터
dataset = load_boston()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=32
)

model = DecisionTreeRegressor(max_depth=4)

model.fit(x_train,y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc :", acc)

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
new_data=[]
feature=[]
for i in range(len(dataset.data[0])):
    if model.feature_importances_[i] !=0:
       new_data.append(df.iloc[:,i])
       feature.append(dataset.feature_names[i])

new_data = pd.concat(new_data, axis=1)
print(new_data.shape)
        
x2_train, x2_test, y2_train, y2_test = train_test_split(new_data, dataset.target, train_size=0.8, random_state=32)
model2 = DecisionTreeRegressor(max_depth=4)   
model2.fit(x2_train,y2_train)
acc2 = model2.score(x2_test, y2_test)
print("acc2 :", acc2)


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

# [0.0621324  0.         0.0078422  0.         0.02805077 0.26905464
#  0.         0.01367513 0.         0.         0.00972972 0.00993
#  0.59958514]
# acc : 0.80463639976642
# (506, 8)
# acc2 : 0.8046363997664201