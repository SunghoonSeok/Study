# a = 1
# b = 2
# c = a + b
# print(c)

# import tensorflow as tensorflow
# import keras
# import numpy as np
# # print("잘 설치됐다.")
# x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])

# x = x[:,:2]
# print(x)
import numpy as np
import pandas as pd
x = np.array(range(9))
print(x)

for i in range(9):
    print('%d'%(i+1))

x = np.array([1,2,3,4,5,6,7,8,9])
print(x.shape)
# df = pd.DataFrame({'cat': ['A','A','A','A','A','B','B','B','B','B','B'],
#                    'sales': [10, 20, 30, 40, 50, 1, 2, 3, 4, 5,6]})
# df['sales'].quantile(q=0.5, interpolation='nearest')

# print(df['sales'].quantile(q=0.5, interpolation='nearest')) # 5
# df.groupby(['cat'])['sales'].quantile(q=0.50, interpolation='nearest')

# print(df.groupby(['cat'])['sales'].quantile(q=0.50, interpolation='nearest'))
# cat
# A    30
# B     3
# Name: sales, dtype: int64
