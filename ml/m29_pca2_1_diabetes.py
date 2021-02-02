import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (442, 10) (442,)

# pca = PCA(n_components=9)
# x2 = pca.fit_transform(x)
# print(x2)
# print(x2.shape) # (442, 7)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR)) 

# 7 : 0.9479436357350414
# 8 : 0.9913119559917797
# 9 : 0.9991439470098977

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print("cumsum :",cumsum)

d = np.argmax(cumsum >=0.95)+1
print("cumsum >=0.95 :", cumsum>=0.95)
print("d :", d)

# cumsum : [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]
# cumsum >=0.95 : [False False False False False False False  True  True  True]
# d : 8