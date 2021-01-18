import numpy as np
import pandas as pd

from pandas import read_csv
df = read_csv('c:/data/test/삼성전자.csv', index_col=0, header=0)


print(df.shape) # (2400, 14)
#Index(['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관',
#       '외인(수량)', '외국계', '프로그램', '외인비']

# str -> int, 불필요한 행 제거
df = df.drop(['2018-05-03','2018-05-02','2018-04-30'])
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
print(df.shape)

# 데이터 추가, str -> float
df2 = read_csv('c:/data/test/삼성전자0115.csv', encoding='cp949', index_col=0, header=0, thousands=',')
df2 = df2.dropna()
df2 = df2.drop(['전일비','Unnamed: 6'], axis=1)

# 중복 데이터 제거
print(df.shape)
df2 = df2.drop(df2.index[3:])
df = df.drop(['2021-01-13'])
print(df2.shape)
df2.index=['2021-01-15','2021-01-14','2021-01-13']

# 액면분할 이전 데이터 주가변환
df = df[::-1]
df.loc[:'2018-05-04','시가':'종가'] = (df.loc[:'2018-05-04','시가':'종가'])/50.
df.loc[:'2018-05-04','거래량'] = (df.loc[:'2018-05-04','거래량'])*50.
df.loc[:'2018-05-04','개인':'프로그램'] = (df.loc[:'2018-05-04','개인':'프로그램'])*50.

# 데이터 병합
df = pd.concat([df, df2], axis=0)
print(df.shape)
print(df.tail())

# 타겟(y값) 설정
dataset_y = df.iloc[:,0]
df['Target'] = dataset_y

# 불필요한 특성 제거
# df = df.drop(['외인비'], axis=1)
# df = df.drop(['개인','기관','외인(수량)','외국계','프로그램'], axis=1)
print(df.shape)

# csv -> npy 후 저장
data = df.values
print(data)
np.save('c:/data/test/samsung_jusik_all.npy', arr=data)


# KODEX data

df3 = read_csv('c:/data/test/KODEX 코스닥150 선물인버스.csv', index_col=0, header=0, encoding='cp949', thousands=',')
df3 = df3.dropna()
df3 = df3.drop(['전일비','Unnamed: 6'], axis=1)
print(df3)

# 데이터 역전
df3 = df3[::-1]

# 불필요한 특성 제거
df3 = df3.drop(['2018/05/03','2018/05/02','2018/04/30'])
# df3 = df3.drop(['외인비'], axis=1)
# df3 = df3.drop(['개인','기관','외인(수량)','외국계','프로그램'], axis=1)
print(df3.shape)

# csv -> npy 후 저장
data2 = df3.values
print(data2)
np.save('c:/data/test/kodex_jusik_all.npy', arr=data2)


# 상관계수
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(style='dark', font_scale=1.2, font='Malgun Gothic') # , palette='pastel'
# sns.color_palette('Paired',6)
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# plt.show()