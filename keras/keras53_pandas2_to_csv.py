import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

x = dataset.data #(150, 4) 
y = dataset.target #(150, )

df = pd.DataFrame(x, columns=dataset.feature_names, index=range(1,151))

df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# y칼럼 추가
df['Target'] = y

df.to_csv('../data/csv/iris_sklearn.csv', sep=',')