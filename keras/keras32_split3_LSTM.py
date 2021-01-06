# 과제 및 실습 Dense으로
# 전처리, es 등등 다 넣어라
# 데이터 1 ~ 100 / 5개씩
#      x           y
# 1,2,3,4,5        6
# ...
# 95,96,97,98,99   100
# predict를 만들것
# 96,97,98,99,100 -> 101
# ...
# 100,101,102,103,104 -> 105
# 예상 predict는 (101, 102, 103, 104, 105)

import numpy as np
from numpy import array

a = np.array(range(1,101))
size = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])  # [subset]과의 차이는?
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)

x = dataset[:, :-1]
y = dataset[:, -1]

pred = split_x(range(96,106),6)
x_pred = pred[:, :-1]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(40, activation='relu', input_shape=(5,1))) #(행, 열, 몇개씩자르는지)
model.add(Dense(60))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(1))

model.summary() 

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=2000, batch_size=16, callbacks=[early_stopping], validation_data=(x_val, y_val))



# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=16)



x_pred = x_pred.reshape(5,5,1)
y_pred = model.predict(x_pred)
print('loss : ',loss)
print(y_pred)


# LSTM
# loss :  0.018539389595389366
# [[101.55333]
#  [102.64213]
#  [103.73669]
#  [104.837  ]
#  [105.94312]]

# loss :  0.0019484958611428738
# [[101.291275]
#  [102.35967 ]
#  [103.43354 ]
#  [104.51292 ]
#  [105.59781 ]]