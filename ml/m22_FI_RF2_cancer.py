from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# 1. 데이터
dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=32
)

model = RandomForestClassifier(max_depth=4)

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
model2 = RandomForestClassifier(max_depth=4)   
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

# [0.05409088 0.00927201 0.06909376 0.03297716 0.00432378 0.01777356 
#  0.03798771 0.07940953 0.00271488 0.00185409 0.01054113 0.00247811 
#  0.01312418 0.05116761 0.00242021 0.00284654 0.00280703 0.00141989
#  0.00308387 0.00199157 0.14726855 0.01043686 0.11409174 0.12774256
#  0.01226258 0.01899609 0.03274262 0.12651601 0.00465891 0.00390656]
# acc : 0.9473684210526315
# acc2 : 0.956140350877193
# (569, 22)
# [0.05759264 0.00941242 0.06036764 0.03957434 0.00628473 0.00874193
#  0.04043355 0.06998312 0.02086387 0.00583402 0.03333898 0.00486034
#  0.14138874 0.01272346 0.2052773  0.12123088 0.01266767 0.00946129
#  0.05307352 0.07335893 0.0066663  0.00686432]