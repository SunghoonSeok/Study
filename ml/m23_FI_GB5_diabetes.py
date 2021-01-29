from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# 1. 데이터
dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=32
)

model = GradientBoostingRegressor(max_depth=4)

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
model2 = GradientBoostingRegressor(max_depth=4)   
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

# [0.05913616 0.01394035 0.21968298 0.09714897 0.03812749 0.06213769
#  0.04387895 0.01743698 0.38063109 0.06787935]
# acc : 0.33316707844223004
# acc2 : 0.3241320062202243
# (442, 7)