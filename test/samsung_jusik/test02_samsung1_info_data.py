import numpy as np
import pandas as pd
from pandas import read_csv
df = read_csv('c:/data/test/삼성전자.csv', index_col=0, header=0) # header가 없으면 none이라 해준다.

print(df.shape) # (2400, 14)
print(df.columns) #Index(['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관',
#       '와인(수량)', '외국계', '프로그램', '외인비'],
print(df.info()) # <class 'pandas.core.frame.DataFrame'>
print(df.describe())

# Data columns (total 14 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   시가      2400 non-null   object
#  1   고가      2400 non-null   object
#  2   저가      2400 non-null   object
#  3   종가      2400 non-null   object
#  4   등락률     2400 non-null   float64
#  5   거래량     2397 non-null   object
#  6   금액(백만)  2397 non-null   object
#  7   신용비     2400 non-null   float64
#  8   개인      2400 non-null   object
#  9   기관      2400 non-null   object
#  10  와인(수량)  2400 non-null   object
#  11  외국계     2400 non-null   object
#  12  프로그램    2400 non-null   object
#  13  외인비     2400 non-null   float64
# dtypes: float64(3), object(11)
# memory usage: 281.2+ KB
# None
print(df.tail())
print(df.isnull()) # False -> 다 차있다
print(df.isnull().sum()) # 차있는 곳의 합

df = df[::-1]
print(df.tail())
# pandas를 넘파이로 바꾸는 것을 찾아라.
samsung = df.to_numpy() # dtype=
print(samsung) # numpy는 데이터 타입을 하나로 통일해야함 그래서 target도 float형이 되었다.



# np.save('c:/data/test/samsung_data.npy', arr=samsung)