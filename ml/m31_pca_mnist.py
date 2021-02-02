import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)
x = x.reshape(70000, 28*28)
# 실습
# pca를 통해 0.95 이상인거 몇개?
# pca 배운거 다 집어넣고 확인!!

pca = PCA()
x = pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print("cumsum :",cumsum)

d = np.argmax(cumsum >=0.95)+1
print("cumsum >=0.95 :", cumsum>=0.95)
print("d :", d) # d : 674 714  154

