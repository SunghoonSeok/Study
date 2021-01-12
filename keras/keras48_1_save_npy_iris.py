from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
print(dataset)
print(dataset.keys()) # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])


x_data = dataset.data
y_data = dataset.target
# x_data = dataset['data']
# y_data = dataset['target']
print(dataset.frame)
print(dataset.DESCR)
print(dataset.feature_names)
print(dataset.target_names) # ['setosa' 'versicolor' 'virginica']
print(dataset.filename) # C:\Users\ai\Anaconda3\lib\site-packages\sklearn\datasets\data\iris.csv

print(type(x_data), type(y_data))

np.save('../data/npy/iris_x.npy', arr=x_data)
np.save('../data/npy/iris_y.npy', arr=y_data)

