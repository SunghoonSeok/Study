# ensemble lstm 과 dense의 조합
# lstm 함수는 저장된 함수 불러오기
# r2구하기
# predict값도 함께 구하기
# lstm data = range(1,510) size=5
# dense data = boston housing
# data 전처리(), early_stopping, hist로 그래프 출력

import numpy as np
from numpy import array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

# 1. 데이터
data = array(range(1,511))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    return array(aaa)

dataset1 = split_x(data, size)
x1 = dataset1[:,:-1]
y1 = dataset1[:,-1]


from sklearn.datasets import load_boston
dataset2 = load_boston()
x2 = dataset2.data
y2 = dataset2.target

print(x1.shape)
print(x2.shape)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=True, random_state=66)
x1_train, x1_val, y1_train, y1_val = train_test_split(x1_train, y1_train, train_size=0.8, shuffle=True)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=True, random_state=66)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=True, random_state=66)
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
scaler1.fit(x1)

