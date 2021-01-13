import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()
print(dataset.keys())
print(dataset.values())

print(dataset.target_names) # ['setosa' 'versicolor' 'virginica']
x = dataset.data #(150, 4) 
y = dataset.target #(150, )
print(type(x), type(y)) # <class 'numpy.ndarray'>

df = pd.DataFrame(x, columns=dataset.feature_names, index=range(1,151))
print(df)
print(df.shape)
print(df.columns)
print(df.index) # 명시 안해주면 자동으로 0부터 인덱싱 해준다

print(df.head()) # df[:5]
print(df.tail()) # df[-5:]
print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 150 entries, 1 to 150
# Data columns (total 4 columns):
#  #   Column             Non-Null Count  Dtype
# ---  ------             --------------  -----
#  0   sepal length (cm)  150 non-null    float64     non-null -> 채워진곳
#  1   sepal width (cm)   150 non-null    float64
#  2   petal length (cm)  150 non-null    float64
#  3   petal width (cm)   150 non-null    float64
# dtypes: float64(4)
# memory usage: 4.8 KB
# None   
print(df.describe()) # 개수, 평균, std, min, max, 25%, 50%, 75%의 정보 담김

df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print(df.columns)
print(df.info())
print(df.describe())

# y칼럼 추가
print(df['sepal_length'])
df['Target'] = dataset.target
print(df.head())

print(df.shape) #(150, 4)
print(df.columns)
print(df.tail())

print(df.info())
print(df.isnull()) # False -> 다 차있다
print(df.isnull().sum()) # 차있는 곳의 합
print(df.describe())
print(df['Target'].value_counts())

# 상관계수, 히트맵
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='dark', font_scale=1.2) # , palette='pastel'
sns.color_palette('Paired',6)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()

# 도수 분포도
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.hist(x='sepal_length', data=df)
plt.title('sepal_length')
# plt.grid()

plt.subplot(2,2,2)
plt.hist(x='sepal_width', data=df)
plt.title('sepal_width')
# plt.grid()

plt.subplot(2,2,3)
plt.hist(x='petal_length', data=df)
plt.title('petal_length')
# plt.grid()

plt.subplot(2,2,4)
plt.hist(x='petal_width', data=df)
plt.title('petal_width')
# plt.grid()


plt.show()

