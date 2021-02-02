import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.
import pandas as pd

dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (442, 10) (442,)


pca = PCA(n_components=10)
x2 = pca.fit_transform(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum :",cumsum)

# d = np.argmax(cumsum >=0.95)+1
# print("cumsum >=0.95 :", cumsum>=0.95)
# print("d :", d)

x_train, x_test, y_train, y_test = train_test_split(
    x2, y, train_size=0.8, random_state=32
)


model = RandomForestRegressor(max_depth=4)

model.fit(x_train,y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print(pca.explained_variance_ratio_)
print("acc :", acc)

# cumsum : [0.80582318 0.96887514 0.99022375 0.99718074 0.99848069 0.99920791
#  0.99962696 0.9998755  0.99996089 0.9999917  0.99999835 0.99999992
#  1.        ]
# cumsum >=0.95 : [False  True  True  True  True  True  True  True  True  True  True  True
#   True]
# d : 2
# acc : 0.8437207800409394

# [8.05823175e-01 1.63051968e-01 2.13486092e-02 6.95699061e-03
#  1.29995193e-03 7.27220158e-04 4.19044539e-04 2.48538539e-04
#  8.53912023e-05 3.08071548e-05]
# acc : 0.6630796775827068