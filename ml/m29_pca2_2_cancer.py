import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (569, 30) (569,)

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

d = np.argmax(cumsum >=0.99)+1
print("cumsum >=0.99 :", cumsum>=0.99)
print("d :", d)

# cumsum : [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999 1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.        ]
# cumsum >=0.99 : [False  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True]
# d : 2