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

# [1.37762725e-05 2.57058514e-02 1.38979084e-05 1.50201789e-05
#  2.20130131e-04 4.30855580e-04 1.64698904e-03 3.29906665e-03
#  9.39461316e-04 1.02311684e-06 7.45933786e-04 3.40707351e-03
#  3.94395544e-03 1.07075298e-02 1.53680838e-04 4.54578256e-04
#  8.84317754e-04 6.92857976e-04 7.39868834e-03 3.61628335e-03
#  7.44692789e-01 4.21540578e-02 1.20141038e-02 1.40836538e-02
#  7.63055984e-03 3.06567442e-03 2.19293656e-02 8.46063586e-02
#  2.91766795e-03 2.61479815e-03]
# acc : 0.9473684210526315
# acc2 : 0.9649122807017544
# (569, 22)