import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (442, 10) (442,)


pca = PCA(n_components=8)
x2 = pca.fit_transform(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print("cumsum :",cumsum)

d = np.argmax(cumsum >=0.95)+1
print("cumsum >=0.95 :", cumsum>=0.95)
print("d :", d)

x_train, x_test, y_train, y_test = train_test_split(
    x2, y, train_size=0.8, random_state=32
)

model = RandomForestRegressor(max_depth=4)

model.fit(x_train,y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print("acc :", acc)

# cumsum : [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196]
# cumsum >=0.95 : [False False False False False False False  True]
# d : 8
# [0.51955082 0.07305534 0.07919319 0.22096223 0.01626107 0.01781968
#  0.05693691 0.01622077]
# acc : 0.4713125962677296