from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# 1. 데이터
dataset = load_breast_cancer()
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
print(model2.feature_importances_)

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

# [2.72097176e-07 2.47922082e-02 7.23164747e-05 2.26051574e-05
#  1.78999348e-04 1.62814053e-03 2.42466078e-03 3.39215704e-03
#  1.93564487e-03 5.25665555e-05 1.11948948e-03 2.46870414e-03
#  4.47727709e-03 1.12239204e-02 1.59158473e-06 1.36755610e-03
#  5.11515989e-04 7.01454866e-04 5.47115683e-03 2.71584371e-03
#  7.45218221e-01 4.24336398e-02 1.18412313e-02 1.36989173e-02
#  7.44651371e-03 2.20042825e-03 2.16296484e-02 8.38812371e-02
#  4.15103567e-03 2.94104616e-03]
# acc : 0.9473684210526315
# acc2 : 0.9473684210526315
# (569, 22)
# [2.63303233e-02 4.84051268e-04 4.30450760e-04 5.97196208e-03
#  3.07922320e-04 7.39038408e-03 3.77498194e-04 4.68459742e-03
#  4.57008862e-03 2.19247799e-03 8.22825782e-03 4.00138486e-03
#  7.45128039e-01 4.27857313e-02 1.19928562e-02 1.35637211e-02
#  7.34846845e-03 3.68563831e-03 2.27957918e-02 8.37510017e-02
#  2.84797741e-03 1.13137550e-03]