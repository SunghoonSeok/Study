import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) # (442, 10) (442,)

pca = PCA(n_components=9)
x2 = pca.fit_transform(x)
print(x2)
print(x2.shape) # (442, 7)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR)) 

# 7 : 0.9479436357350414
# 8 : 0.9913119559917797
# 9 : 0.9991439470098977