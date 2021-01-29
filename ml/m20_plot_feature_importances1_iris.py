from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 1. 데이터
dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=32
)

model = DecisionTreeClassifier(max_depth=4)

model.fit(x_train,y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc :", acc)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

# [0.01271995 0.         0.54468575 0.4425943 ]
# acc : 0.9666666666666667

# [0.         0.01271995 0.04924373 0.93803632]
# acc : 1.0