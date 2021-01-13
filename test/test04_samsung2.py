import numpy as np
import pandas as pd

from pandas import read_csv
df = read_csv('./test/삼성전자.csv', index_col=0, header=0)

# print(df.shape) # (2400, 14)
# print(df.columns) #Index(['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관',
# #       '와인(수량)', '외국계', '프로그램', '외인비'],
df = df.fillna(method='pad')
df['시가'] =df['시가'].str.replace(',','').astype('int64')
df['고가'] =df['고가'].str.replace(',','').astype('int64')
df['저가'] =df['저가'].str.replace(',','').astype('int64')
df['종가'] =df['종가'].str.replace(',','').astype('int64')
df['거래량'] =df['거래량'].str.replace(',','').astype('int64')
df['금액(백만)'] =df['금액(백만)'].str.replace(',','').astype('int64')
df['개인'] =df['개인'].str.replace(',','').astype('int64')
df['기관'] =df['기관'].str.replace(',','').astype('int64')
df['외인(수량)'] =df['외인(수량)'].str.replace(',','').astype('int64')
df['외국계'] =df['외국계'].str.replace(',','').astype('int64')
df['프로그램'] =df['프로그램'].str.replace(',','').astype('int64')

df = df[::-1]
# print(df.iloc[0:3,0:5])
print(df.iloc[-662:,:])

# dataset_x = df.iloc[-662:,:]
dataset_y = df.iloc[:,3]


df['Target'] = dataset_y
print(df)
data = df.iloc[-662:,:].values
print(data)
np.save('./test/samsung_data.npy', arr=data)
# print(dataset_x)
# print(dataset_y)

# np.save('./test/samsung_data_x.npy', arr=dataset_x)
# np.save('./test/samsung_data_y.npy', arr=dataset_y)
