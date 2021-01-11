'''
훈련1
앙상블 x2개 y3개
diabetes 데이터  cnn으로
diabetes와 데이터개수가 동일한 정수 시계열 데이터 생성 lstm으로, y값2개
전처리, es, dropout, maxpooling 사용
test값을 이용한 predict값 도출 rmse,r2도 도출
만약 도움받은 부분 있으면 주석에 표시할것
'''
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Input, LSTM, MaxPooling2D, Dropout, Flatten

# 1. 데이터
from sklearn.datasets import load_diabetes
dataset1 = load_diabetes()
x1 = dataset1.data
y1 = dataset1.target
print(x1.shape, y1.shape) #(442, 10), (442,)

data2 = np.array(range(1, 450))
size = 8
def split_x(seq, size): # 도움받음 ㅠ
    aaa=[]
    for i in range(len(seq)-size+1): # i를 range len(seq)-size+1
        subset = seq[i : (i+size)] # seq의 [0:8]을 가져온다 -> [1,2,3,4,5,6,7,8] .....
        aaa.append([item for item in subset]) # aaa가 subset안에 있는 요소들을 그대로 return해서 aaa안에 집어 넣는다
    return np.array(aaa)
dataset2 = split_x(data2, size)
x2 = dataset2[:,:-3]
y2 = dataset2[:,-3:-1]
y3 = dataset2[:,-1]

print(x2.shape, y2.shape, y3.shape) # (442, 5) (442, 2) (442,)

