import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0) # header가 없으면 none이라 해준다.
print(df)

print(df.shape) #(150, 5)
print(df.info())

# pandas를 넘파이로 바꾸는 것을 찾아라.
aaa = df.to_numpy() # dtype=
print(aaa) # numpy는 데이터 타입을 하나로 통일해야함 그래서 target도 float형이 되었다.
bbb = df.values
print(bbb)

np.save('../data/npy/iris_sklearn.npy', arr=aaa)

# 과제
# pandas의 loc iloc에 대해 정리

