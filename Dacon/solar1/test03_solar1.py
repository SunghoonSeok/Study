# data 로드하기
# index = None

import numpy as np
import pandas as pd

# 데이터 로드
from pandas import read_csv
df = read_csv('c:/data/test/solar/train/train.csv', index_col=None, header=0)
print(df.tail())

data = df.values
print(data.shape)
np.save('c:/data/test/solar/train.npy', arr=data)
